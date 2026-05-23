# llama-vortex

A custom Vortex RISC-V GPGPU backend for [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling LLM inference on the open-source [Vortex](https://github.com/vortexgpgpu/vortex) GPU architecture via the SimX simulator.

## Overview

This project ports llama.cpp to run on Vortex, a fully open-source RISC-V GPGPU developed at Georgia Tech. Instead of relying on proprietary ecosystems like NVIDIA CUDA or Apple Metal, this backend allows LLM inference to run on an open, research-friendly hardware platform.

The backend is validated using **SimX**, Vortex's cycle-accurate C++ simulator, so no physical FPGA hardware is required.

### Architecture

```
llama.cpp (inference engine)
    └── ggml tensor library
            └── ggml-vortex backend  ← this repo
                    └── SimX (Vortex RISC-V GPU simulator)
```

### Current Status

- **Phase 1 (complete):** Full backend infrastructure implemented. Inference pipeline runs end-to-end through the Vortex backend with CPU fallback.
- **Phase 2 (in progress):** Replace CPU fallback with real OpenCL kernels dispatched to SimX (matmul, RMSNorm, Attention).

---

## Repository Structure

```
llama-vortex/
├── ggml-vortex/
│   ├── ggml-vortex.cpp        # Vortex backend implementation
│   └── CMakeLists.txt         # Build config for the backend
├── ggml-vortex.h              # Public API header
├── ggml-backend-reg.cpp       # Modified: registers Vortex into llama.cpp backend system
├── ggml-src-CMakeLists.txt    # Modified: adds Vortex as a build option
└── ggml-CMakeLists.txt        # Modified: adds GGML_VORTEX cmake option
```

---

## Setup & Validation

### Prerequisites

- Ubuntu 22.04 (use Docker if on macOS — Vortex toolchain is Linux-only)
- cmake >= 3.14
- git

### Step 1: Set up the environment (Docker recommended on macOS)

```bash
docker run -dit --name vortex ubuntu:22.04 bash
docker exec -it vortex bash
apt-get update && apt-get install -y git cmake build-essential wget
```

### Step 2: Clone llama.cpp

```bash
cd /home
git clone --depth=1 https://github.com/ggml-org/llama.cpp
```

### Step 3: Apply the Vortex backend

Clone this repo and copy the files into llama.cpp:

```bash
git clone https://github.com/mizhao34-hash/llama-vortex
cd llama-vortex

cp -r ggml-vortex         ../llama.cpp/ggml/src/ggml-vortex
cp ggml-vortex.h          ../llama.cpp/ggml/include/ggml-vortex.h
cp ggml-backend-reg.cpp   ../llama.cpp/ggml/src/ggml-backend-reg.cpp
cp ggml-src-CMakeLists.txt ../llama.cpp/ggml/src/CMakeLists.txt
cp ggml-CMakeLists.txt     ../llama.cpp/ggml/CMakeLists.txt
```

### Step 4: Build llama.cpp with Vortex backend

```bash
cd ../llama.cpp
cmake -B build -DGGML_VORTEX=ON
cmake --build build --config Release -j4
```

### Step 5: Verify the Vortex device is recognized

```bash
./build/bin/llama-cli --list-devices
```

Expected output:
```
Available devices:
  Vortex: Vortex RISC-V GPGPU (SimX) (1024 MiB, 1024 MiB free)
```

### Step 6: Run inference with the Vortex backend

Download a small test model and run:

```bash
wget https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
./build/bin/llama-cli -m stories260K.gguf -p "Once upon a time" -n 50 --device Vortex
```

Expected output includes:
```
[Vortex] backend initialized
[Vortex] graph_compute: N nodes (CPU fallback)
```

---

## Implementation Details

### Files Modified in llama.cpp

| File | Change |
|------|--------|
| `ggml/src/CMakeLists.txt` | Added `ggml_add_backend(Vortex)` |
| `ggml/CMakeLists.txt` | Added `option(GGML_VORTEX ...)` |
| `ggml/src/ggml-backend-reg.cpp` | Added static registration of Vortex backend |

### Files Added

| File | Description |
|------|-------------|
| `ggml/src/ggml-vortex/ggml-vortex.cpp` | Full backend implementation: GUID, buffer type, device interface, graph compute, registry |
| `ggml/src/ggml-vortex/CMakeLists.txt` | Build configuration for the Vortex backend library |
| `ggml/include/ggml-vortex.h` | Public API: `ggml_backend_vortex_init()`, `ggml_backend_vortex_reg()` |

### Backend Interface

The backend implements the full `ggml_backend_i` interface required by llama.cpp:

- **Buffer type:** allocates host-accessible memory (32-byte aligned for RISC-V)
- **Device interface:** exposes Vortex as a GPU-type device with 1GB memory
- **Graph compute:** currently delegates to CPU (Phase 1); will dispatch to SimX OpenCL kernels in Phase 2
- **Registry:** registers the backend so it is auto-discovered at startup

---

## Roadmap

- [x] Environment setup (Docker + Ubuntu 22.04)
- [x] Build Vortex + SimX simulator
- [x] Build llama.cpp with CPU backend
- [x] Implement ggml-vortex backend infrastructure
- [x] Register Vortex as selectable device (`--device Vortex`)
- [x] Verify end-to-end inference pipeline
- [ ] Write matmul OpenCL kernel for SimX
- [ ] Replace CPU fallback with real SimX dispatch
- [ ] Implement RMSNorm and Attention kernels
- [ ] Run TinyLlama Q4 benchmark on SimX
- [ ] Measure latency and memory bandwidth

---

## Authors

- Mingqi Zhao
- Jaelyn Fan

Course project for CS254A, UCLA