#include "ggml-vortex.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>

// Vortex backend for llama.cpp
// Phase 2: matmul dispatched to Vortex/SimX via OpenCL
// Other ops fall back to CPU

// --- OpenCL state ---

struct vortex_ocl_context {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       ctx;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        matmul_kernel;
    bool             initialized;
};

static struct vortex_ocl_context ocl = {0};

static const char * matmul_kernel_src =
"__kernel void matmul(__global const float *A,\n"
"                     __global const float *B,\n"
"                     __global float *C,\n"
"                     int N)\n"
"{\n"
"    const int row = get_global_id(0);\n"
"    const int col = get_global_id(1);\n"
"    float acc = 0.0f;\n"
"    for (int k = 0; k < N; k++) {\n"
"        acc += A[k * N + row] * B[col * N + k];\n"
"    }\n"
"    C[col * N + row] = acc;\n"
"}\n";

static bool vortex_ocl_init(void) {
    if (ocl.initialized) return true;

    cl_int err;
    err = clGetPlatformIDs(1, &ocl.platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[Vortex] clGetPlatformIDs failed: %d\n", err);
        return false;
    }
    err = clGetDeviceIDs(ocl.platform, CL_DEVICE_TYPE_DEFAULT, 1, &ocl.device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[Vortex] clGetDeviceIDs failed: %d\n", err);
        return false;
    }
    ocl.ctx = clCreateContext(NULL, 1, &ocl.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[Vortex] clCreateContext failed: %d\n", err);
        return false;
    }
    ocl.queue = clCreateCommandQueue(ocl.ctx, ocl.device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[Vortex] clCreateCommandQueue failed: %d\n", err);
        return false;
    }

    ocl.initialized = true;
    fprintf(stderr, "[Vortex] OpenCL context initialized\n");
    return true;
}

// Run matmul on Vortex OpenCL: C = A * B (square NxN, float32)
static bool vortex_matmul_rect(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!vortex_ocl_init()) return false;

    // Update kernel source to handle rectangular matrices
    static const char * rect_kernel_src =
        "__kernel void matmul_rect(__global const float *A,\n"
        "                          __global const float *B,\n"
        "                          __global float *C,\n"
        "                          int M, int N, int K)\n"
        "{\n"
        "    const int row = get_global_id(0);\n"
        "    const int col = get_global_id(1);\n"
        "    if (row >= M || col >= N) return;\n"
        "    float acc = 0.0f;\n"
        "    for (int k = 0; k < K; k++) {\n"
        "        acc += A[row * K + k] * B[k * N + col];\n"
        "    }\n"
        "    C[row * N + col] = acc;\n"
        "}\n";

    cl_int err;

    // Build rect kernel if not done yet
    static cl_kernel rect_kernel = NULL;
    if (!rect_kernel) {
        cl_program prog = clCreateProgramWithSource(ocl.ctx, 1, &rect_kernel_src, NULL, &err);
        if (err != CL_SUCCESS) return false;
        err = clBuildProgram(prog, 1, &ocl.device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(prog, ocl.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char*)malloc(log_size);
            clGetProgramBuildInfo(prog, ocl.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "[Vortex] Build log (err=%d, log_size=%zu):\n%s\n", err, log_size, log);
            free(log);
            clReleaseProgram(prog);
            return false;
        }
        rect_kernel = clCreateKernel(prog, "matmul_rect", &err);
        if (err != CL_SUCCESS) return false;
    }

    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    cl_mem a_buf = clCreateBuffer(ocl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_bytes, (void*)A, &err);
    if (err != CL_SUCCESS) return false;
    cl_mem b_buf = clCreateBuffer(ocl.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_bytes, (void*)B, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(a_buf); return false; }
    cl_mem c_buf = clCreateBuffer(ocl.ctx, CL_MEM_WRITE_ONLY, c_bytes, NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(a_buf); clReleaseMemObject(b_buf); return false; }

    clSetKernelArg(rect_kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(rect_kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(rect_kernel, 2, sizeof(cl_mem), &c_buf);
    clSetKernelArg(rect_kernel, 3, sizeof(int), &M);
    clSetKernelArg(rect_kernel, 4, sizeof(int), &N);
    clSetKernelArg(rect_kernel, 5, sizeof(int), &K);

    size_t global[2] = {(size_t)M, (size_t)N};
    size_t local[2]  = {1, 1};
    fprintf(stderr, "[Vortex] launching kernel M=%d N=%d K=%d\n", M, N, K);
    err = clEnqueueNDRangeKernel(ocl.queue, rect_kernel, 2, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[Vortex] enqueue failed: %d\n", err);
        clReleaseMemObject(a_buf); clReleaseMemObject(b_buf); clReleaseMemObject(c_buf);
        return false;
    }
    clFinish(ocl.queue);
    fprintf(stderr, "[Vortex] kernel finished\n");
    clEnqueueReadBuffer(ocl.queue, c_buf, CL_TRUE, 0, c_bytes, C, 0, NULL, NULL);

    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    return true;
}

// --- GUID ---

static ggml_guid vortex_guid = {
    0x76, 0x6f, 0x72, 0x74, 0x65, 0x78, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
};
ggml_guid_t ggml_backend_vortex_guid(void) { return &vortex_guid; }

// --- Buffer type ---

static const char * ggml_backend_vortex_buffer_type_name(ggml_backend_buffer_type_t buft) {
    (void)buft; return "Vortex";
}
static ggml_backend_buffer_t ggml_backend_vortex_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    (void)buft;
    return ggml_backend_cpu_buffer_type()->iface.alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}
static size_t ggml_backend_vortex_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft; return 32;
}
static bool ggml_backend_vortex_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft; return true;
}
static struct ggml_backend_buffer_type_i vortex_buffer_type_interface = {
    /* .get_name      = */ ggml_backend_vortex_buffer_type_name,
    /* .alloc_buffer  = */ ggml_backend_vortex_buffer_type_alloc_buffer,
    /* .get_alignment = */ ggml_backend_vortex_buffer_type_get_alignment,
    /* .get_max_size  = */ NULL,
    /* .get_alloc_size= */ NULL,
    /* .is_host       = */ ggml_backend_vortex_buffer_type_is_host,
};
static struct ggml_backend_buffer_type vortex_buffer_type = {
    /* .iface   = */ vortex_buffer_type_interface,
    /* .device  = */ NULL,
    /* .context = */ NULL,
};
ggml_backend_buffer_type_t ggml_backend_vortex_buffer_type(void) {
    return &vortex_buffer_type;
}

// --- Device ---

static const char * ggml_backend_vortex_dev_get_name(ggml_backend_dev_t dev) {
    (void)dev; return "Vortex";
}
static const char * ggml_backend_vortex_dev_get_description(ggml_backend_dev_t dev) {
    (void)dev; return "Vortex RISC-V GPGPU (SimX)";
}
static void ggml_backend_vortex_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    *free  = 1ULL * 1024 * 1024 * 1024;
    *total = 1ULL * 1024 * 1024 * 1024;
}
static enum ggml_backend_dev_type ggml_backend_vortex_dev_get_type(ggml_backend_dev_t dev) {
    (void)dev; return GGML_BACKEND_DEVICE_TYPE_GPU;
}
static void ggml_backend_vortex_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_vortex_dev_get_name(dev);
    props->description = ggml_backend_vortex_dev_get_description(dev);
    props->type        = ggml_backend_vortex_dev_get_type(dev);
    ggml_backend_vortex_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {false, false, false, false};
}
static ggml_backend_t ggml_backend_vortex_dev_init_backend(ggml_backend_dev_t dev, const char * params);
static ggml_backend_buffer_type_t ggml_backend_vortex_dev_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev; return ggml_backend_vortex_buffer_type();
}
static bool ggml_backend_vortex_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev; (void)op; return true;
}
static bool ggml_backend_vortex_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    // Accept both Vortex and CPU buffers so we can access model weights
    return buft == ggml_backend_vortex_buffer_type() ||
           buft == ggml_backend_cpu_buffer_type();
}
static struct ggml_backend_device_i vortex_device_interface = {
    /* .get_name             = */ ggml_backend_vortex_dev_get_name,
    /* .get_description      = */ ggml_backend_vortex_dev_get_description,
    /* .get_memory           = */ ggml_backend_vortex_dev_get_memory,
    /* .get_type             = */ ggml_backend_vortex_dev_get_type,
    /* .get_props            = */ ggml_backend_vortex_dev_get_props,
    /* .init_backend         = */ ggml_backend_vortex_dev_init_backend,
    /* .get_buffer_type      = */ ggml_backend_vortex_dev_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_vortex_dev_supports_op,
    /* .supports_buft        = */ ggml_backend_vortex_dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
static struct ggml_backend_device vortex_device = {
    /* .iface   = */ vortex_device_interface,
    /* .reg     = */ NULL,
    /* .context = */ NULL,
};

// --- Backend ---

static const char * ggml_backend_vortex_get_name(ggml_backend_t backend) {
    (void)backend; return "Vortex";
}
static void ggml_backend_vortex_free(ggml_backend_t backend) { free(backend); }

static enum ggml_status ggml_backend_vortex_graph_compute(
        ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    (void)backend;

    int vortex_ops = 0;
    int cpu_ops    = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // Dispatch square float32 matmul to Vortex OpenCL
        if (node->op == GGML_OP_MUL_MAT) {
            fprintf(stderr, "[Vortex] MUL_MAT: src0 type=%d shape=[%ld,%ld], src1 type=%d shape=[%ld,%ld]\n",
            node->src[0]->type, node->src[0]->ne[0], node->src[0]->ne[1],
            node->src[1]->type, node->src[1]->ne[0], node->src[1]->ne[1]);
            }
        if (node->op == GGML_OP_MUL_MAT &&
            node->src[0]->type == GGML_TYPE_F32 &&
            node->src[1]->type == GGML_TYPE_F32) {

            int64_t M = node->src[0]->ne[1]; // rows of A
            int64_t K = node->src[0]->ne[0]; // cols of A = rows of B
            int64_t N = node->src[1]->ne[1]; // cols of B

            const float * A = (const float *)node->src[0]->data;
            const float * B = (const float *)node->src[1]->data;
            float       * C = (float *)node->data;

            if (vortex_matmul_rect(A, B, C, (int)M, (int)N, (int)K)) {
                fprintf(stderr, "[Vortex] matmul %ldx%ldx%ld dispatched to OpenCL\n", M, K, N);
                vortex_ops++;
                continue;
            }
        }
        else if (node->op == GGML_OP_MUL_MAT &&
            node->src[0]->type == GGML_TYPE_Q4_0 &&
            node->src[1]->type == GGML_TYPE_F32) {

            int64_t M = node->src[0]->ne[1];
            int64_t K = node->src[0]->ne[0];
            int64_t N = node->src[1]->ne[1];

            // dequantize Q4_0 weights to F32 first
            int64_t n_elems = M * K;
            float * A_f32 = (float *)malloc(n_elems * sizeof(float));
            if (A_f32) {
                // each block_q4_0 has 32 elements
                const int QK4_0 = 32;
                int64_t n_blocks = n_elems / QK4_0;
                const uint8_t * qdata = (const uint8_t *)node->src[0]->data;

                for (int64_t b = 0; b < n_blocks; b++) {
                    // read scale (fp16 -> fp32)
                    uint16_t d_bits;
                    memcpy(&d_bits, qdata + b * (2 + QK4_0/2), 2);
                    // simple fp16 to fp32 conversion
                    uint32_t exp = (d_bits >> 10) & 0x1F;
                    uint32_t mant = d_bits & 0x3FF;
                    uint32_t sign = (d_bits >> 15) & 0x1;
                    float d;
                    if (exp == 0) {
                        d = (sign ? -1.0f : 1.0f) * ldexpf(mant / 1024.0f, -14);
                    } else if (exp == 31) {
                        d = mant ? NAN : (sign ? -INFINITY : INFINITY);
                    } else {
                        uint32_t f32bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                        memcpy(&d, &f32bits, 4);
                    }

                    const uint8_t * qs = qdata + b * (2 + QK4_0/2) + 2;
                    for (int j = 0; j < QK4_0/2; j++) {
                        int v0 = (qs[j] & 0x0F) - 8;
                        int v1 = (qs[j] >> 4)  - 8;
                        A_f32[b * QK4_0 + j*2]     = v0 * d;
                        A_f32[b * QK4_0 + j*2 + 1] = v1 * d;
                    }
                }

                const float * B = (const float *)node->src[1]->data;
                float       * C = (float *)node->data;

                if (vortex_matmul_rect(A_f32, B, C, (int)M, (int)N, (int)K)) {
                    fprintf(stderr, "[Vortex] Q4_0 matmul %ldx%ldx%ld dispatched to SimX\n", M, K, N);
                    vortex_ops++;
                    free(A_f32);
                    continue;
                }
                free(A_f32);
            }
        }

        // Fall back to CPU for everything else
        cpu_ops++;
    }

    // Run remaining ops on CPU
    if (cpu_ops > 0) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_graph_compute(cpu, cgraph);
        ggml_backend_free(cpu);
    }

    fprintf(stderr, "[Vortex] graph_compute: %d vortex ops, %d cpu ops\n",
            vortex_ops, cpu_ops);

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i vortex_backend_interface = {
    /* .get_name            = */ ggml_backend_vortex_get_name,
    /* .free                = */ ggml_backend_vortex_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_vortex_graph_compute,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .graph_optimize      = */ NULL,
};

ggml_backend_t ggml_backend_vortex_init(void) {
    ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));
    if (!backend) { fprintf(stderr, "[Vortex] alloc failed\n"); return NULL; }
    backend->guid    = ggml_backend_vortex_guid();
    backend->iface   = vortex_backend_interface;
    backend->device  = &vortex_device;
    backend->context = NULL;
    fprintf(stderr, "[Vortex] backend initialized\n");
    return backend;
}

static ggml_backend_t ggml_backend_vortex_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)dev; (void)params; return ggml_backend_vortex_init();
}

bool ggml_backend_is_vortex(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_vortex_get_name;
}

// --- Registry ---

static const char * ggml_backend_vortex_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg; return "Vortex";
}
static size_t ggml_backend_vortex_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg; return 1;
}
static ggml_backend_dev_t ggml_backend_vortex_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg; (void)index; return &vortex_device;
}
static struct ggml_backend_reg_i vortex_reg_interface = {
    /* .get_name         = */ ggml_backend_vortex_reg_get_name,
    /* .get_device_count = */ ggml_backend_vortex_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vortex_reg_get_device,
    /* .get_proc_address = */ NULL,
};
ggml_backend_reg_t ggml_backend_vortex_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ vortex_reg_interface,
        /* .context     = */ NULL,
    };
    vortex_device.reg = &reg;
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_vortex_reg)