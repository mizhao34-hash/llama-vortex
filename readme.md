# llama-vortex

A custom Vortex RISC-V GPGPU backend for [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling LLM inference on the open-source [Vortex](https://github.com/vortexgpgpu/vortex) GPU architecture via the SimX simulator.

## Result

We successfully ran llama.cpp inference on the Vortex RISC-V GPU simulator (SimX):

```
> Once upon a time
But where if we work to build a sta
[ Prompt: 0.0 t/s | Generation: 0.0 t/s ]
```

The near-zero speed is expected — SimX is a cycle-accurate software simulator, ~1000x slower than real hardware. The important result is that the model **actually generated text with matmul kernels running on the Vortex RISC-V GPU**.

## Overview

This project ports llama.cpp to run on Vortex, a fully open-source RISC-V GPGPU developed at Georgia Tech. Instead of relying on proprietary ecosystems like NVIDIA CUDA or Apple Metal, this backend allows LLM inference to run on an open, research-friendly hardware platform.

The backend is validated using **SimX**, Vortex's cycle-accurate C++ simulator, so no physical FPGA hardware is required.

### Architecture

```
llama.cpp (inference engine)
    └── ggml tensor library
            └── ggml-vortex backend  ← this repo
                    └── Vortex OpenCL runtime (POCL)
                            └── SimX (Vortex RISC-V GPU simulator)
```

### What we implemented

- **Phase 1:** Full backend infrastructure. Registered Vortex as a selectable device in llama.cpp (`--device Vortex`). All ops routed through Vortex backend with CPU fallback.
- **Phase 2:** Real matmul kernel dispatched to Vortex SimX via OpenCL. The kernel is compiled to RISC-V machine code by Vortex's POCL and executed on SimX.

---

## Repository Structure

```
llama-vortex/
├── ggml-vortex/
│   ├── ggml-vortex.cpp        # Vortex backend implementation
│   └── CMakeLists.txt         # Build config
├── ggml-vortex.h              # Public API header
├── ggml-backend-reg.cpp       # Modified: registers Vortex into llama.cpp
├── ggml-src-CMakeLists.txt    # Modified: adds Vortex as a build option
└── ggml-CMakeLists.txt        # Modified: adds GGML_VORTEX cmake option
```

---

## Setup & Validation

### Prerequisites

- Ubuntu 22.04 (use Docker if on macOS — Vortex toolchain is Linux-only)
- cmake >= 3.14, git, wget

### Step 1: Set up the environment (Docker recommended on macOS)

```bash
docker run -dit --name vortex ubuntu:22.04 bash
docker exec -it vortex bash
apt-get update && apt-get install -y git cmake build-essential wget
```

### Step 2: Clone and build Vortex + SimX

```bash
cd /home
git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex
cd vortex
bash ci/install_dependencies.sh
mkdir build && cd build
../configure --xlen=32 --tooldir=$HOME/tools
./ci/toolchain_install.sh --all
make -s
```

Verify SimX works:
```bash
./ci/blackbox.sh --cores=2 --app=vecadd
# Expected: PASSED!
```

### Step 3: Clone llama.cpp

```bash
cd /home
git clone --depth=1 https://github.com/ggml-org/llama.cpp
```

### Step 4: Apply the Vortex backend

```bash
git clone https://github.com/mizhao34-hash/llama-vortex
cd llama-vortex

cp -r ggml-vortex           ../llama.cpp/ggml/src/ggml-vortex
cp ggml-vortex.h            ../llama.cpp/ggml/include/ggml-vortex.h
cp ggml-backend-reg.cpp     ../llama.cpp/ggml/src/ggml-backend-reg.cpp
cp ggml-src-CMakeLists.txt  ../llama.cpp/ggml/src/CMakeLists.txt
cp ggml-CMakeLists.txt      ../llama.cpp/ggml/CMakeLists.txt
```

### Step 5: Build llama.cpp with Vortex backend

```bash
cd ../llama.cpp
apt-get install -y pocl-opencl-icd
cmake -B build -DGGML_VORTEX=ON
cmake --build build --config Release -j4
```

### Step 6: Verify the Vortex device is recognized

```bash
./build/bin/llama-cli --list-devices
```

Expected:
```
Available devices:
  Vortex: Vortex RISC-V GPGPU (SimX) (1024 MiB, 1024 MiB free)
```

### Step 7: Run inference on Vortex SimX

Download a small test model:
```bash
wget https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
```

Run with full Vortex SimX environment (replace `/home/cs254A/vortex` with your Vortex path):
```bash
LD_LIBRARY_PATH=/root/tools/pocl/lib:/home/cs254A/vortex/build/sw/runtime:/root/tools/llvm-vortex/lib \
POCL_VORTEX_XLEN=32 \
LLVM_PREFIX=/root/tools/llvm-vortex \
POCL_VORTEX_BINTOOL="OBJCOPY=/root/tools/llvm-vortex/bin/llvm-objcopy /home/cs254A/vortex/sw/kernel/scripts/vxbin.py" \
POCL_VORTEX_CFLAGS="-march=rv32imaf -mabi=ilp32f --target=riscv32-unknown-elf -O3 -mcmodel=medany \
  --sysroot=/root/tools/riscv32-gnu-toolchain/riscv32-unknown-elf \
  --gcc-toolchain=/root/tools/riscv32-gnu-toolchain \
  -fno-rtti -fno-exceptions -nostartfiles -nostdlib \
  -fdata-sections -ffunction-sections \
  -I/home/cs254A/vortex/build/sw -I/home/cs254A/vortex/build/hw \
  -I/home/cs254A/vortex/sw/kernel/include \
  -DVX_CFG_XLEN=32 -DVX_CFG_XLEN_32 -DNDEBUG -D__VORTEX__ \
  -Xclang -target-feature -Xclang +xvortex \
  -Xclang -target-feature -Xclang +zicond \
  -mllvm -disable-loop-idiom-all" \
POCL_VORTEX_LDFLAGS="-fuse-ld=lld -Wl,-z,norelro \
  -Wl,-Bstatic,--gc-sections,-T/home/cs254A/vortex/sw/kernel/scripts/link32.ld,--defsym=STARTUP_ADDR=0x80000000 \
  /home/cs254A/vortex/build/sw/kernel/libvortex2.a \
  -L/root/tools/libc32/lib -lm -lc \
  /root/tools/libcrt32/lib/baremetal/libclang_rt.builtins-riscv32.a" \
VORTEX_DRIVER=simx \
./build/bin/llama-cli -m stories260K.gguf -p "Once upon a time" -n 20 --device Vortex
```

Expected output:
```
[Vortex] OpenCL context initialized
[Vortex] launching kernel M=64 N=2 K=64
[Vortex] kernel finished
[Vortex] matmul 64x64x2 dispatched to OpenCL
...
> Once upon a time
But where if we work to build a sta
```

**Note:** Generation speed shows ~0 t/s because SimX is a cycle-accurate simulator (~1000x slower than real hardware). This is expected behavior — the model is genuinely running on the RISC-V GPU simulator.

---

## Implementation Details

### Files Modified in llama.cpp

| File | Change |
|------|--------|
| `ggml/src/CMakeLists.txt` | Added `ggml_add_backend(Vortex)` |
| `ggml/CMakeLists.txt` | Added `option(GGML_VORTEX ...)` |
| `ggml/src/ggml-backend-reg.cpp` | Static registration of Vortex backend |

### Files Added

| File | Description |
|------|-------------|
| `ggml/src/ggml-vortex/ggml-vortex.cpp` | Backend: GUID, buffer, device, graph_compute, registry |
| `ggml/src/ggml-vortex/CMakeLists.txt` | Links Vortex POCL and runtime |
| `ggml/include/ggml-vortex.h` | Public API |

### How matmul dispatch works

```
graph_compute(cgraph)
    for each node:
        if node.op == MUL_MAT and type == F32:
            → compile kernel to RISC-V via Vortex POCL
            → upload matrices to SimX memory
            → execute on SimX
            → read result back
        else:
            → CPU fallback
```

---

## Roadmap

- [x] Environment setup (Docker + Ubuntu 22.04)
- [x] Build Vortex + SimX simulator
- [x] Build llama.cpp with CPU backend
- [x] Implement ggml-vortex backend infrastructure
- [x] Register Vortex as selectable device (`--device Vortex`)
- [x] Verify end-to-end inference pipeline (CPU fallback)
- [x] Write matmul OpenCL kernel targeting Vortex RISC-V
- [x] Dispatch matmul to Vortex SimX via OpenCL
- [x] End-to-end LLM inference running on Vortex RISC-V GPU
- [ ] RMSNorm and Attention kernels on SimX
- [ ] TinyLlama Q4 full benchmark

---

## Authors

- Mingqi Zhao
- Jaelyn Fan

Course project for CS254A, UCLA