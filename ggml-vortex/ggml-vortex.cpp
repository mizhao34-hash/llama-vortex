#include "ggml-vortex.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Vortex backend for llama.cpp
// Phase 1: CPU fallback - routes computation through CPU while backend infrastructure is in place
// Phase 2 (TODO): replace graph_compute with real Vortex/SimX OpenCL kernels

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

// --- Device (defined before init so we can take its address) ---

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
    (void)dev; return buft == ggml_backend_vortex_buffer_type();
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
    // TODO: dispatch to Vortex/SimX OpenCL kernels
    // For now, fall back to CPU to verify the pipeline is wired correctly
    fprintf(stderr, "[Vortex] graph_compute: %d nodes (CPU fallback)\n", cgraph->n_nodes);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    enum ggml_status status = cpu->iface.graph_compute(cpu, cgraph);
    ggml_backend_free(cpu);
    (void)backend;
    return status;
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
