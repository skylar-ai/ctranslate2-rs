Pixi Issues
-----------

> You must have an NVIDIA GPU on your machine and you must install the CUDA drivers. The CUDA driver cannot be installed with conda and must be installed on your system using an appropriate installation method. [Reference](https://conda-forge.org/docs/maintainer/knowledge_base/#prerequisites).

# Setup

Move from full `cuda` (`coda-forge`) to minimal dependencies:

```toml
[dependencies]
cuda = ">=12.9,<13.0"
```

```toml
[dependencies]
cuda-version = "12.9.*"
cudnn = "8.*"
cuda-nvcc = "*"
libcublas-dev = "*"
```

# NCCL and MPI > tensor-parallel
[NCCL](https://developer.nvidia.com/nccl)

The `CTranslate2` requires `NCCL` and `MPI` to build (line 500-502).

```CmakeLists.txt
if (WITH_TENSOR_PARALLEL)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
```

# Multiple `cuda.h`

```bash
~/ctranslate2-rs$ find .pixi/envs/default/ -name cuda.h
.pixi/envs/default/x86_64-conda-linux-gnu/sysroot/usr/include/linux/cuda.h
.pixi/envs/default/targets/x86_64-linux/include/cuda.h
.pixi/envs/default/include/hwloc/cuda.h
```

Those three `cuda.h` files are **not** duplicate installations of NVIDIA CUDA, and it is **not a bug in conda-forge**. 

### Breakdown of the 3 files

| File Path | Package | What it actually is |
| :--- | :--- | :--- |
| `sysroot/usr/include/linux/cuda.h` | `sysroot_linux-64` | **Linux kernel header for Apple hardware (1996)** |
| `include/hwloc/cuda.h` | `openmpi` / `libhwloc` | **Hardware Locality helper header** |
| `targets/x86_64-linux/include/cuda.h` | `cuda-cudart-dev` | **The real NVIDIA CUDA header** |

### Why does conda-forge put it in `targets/x86_64-linux/`?

Starting with CUDA 11 and 12, NVIDIA distributes the CUDA Toolkit as modular redistributable packages. NVIDIA uses a target-architecture layout to support cross-compilation:

```text
$CONDA_PREFIX/targets/
  └── x86_64-linux/
      ├── include/     <-- cuda.h, cuda_runtime.h, etc.
      └── lib/         <-- libcudart.so, etc.
```

Conda-forge packages NVIDIA's official redistributables directly using NVIDIA's layout. To make compilers find it, conda-forge's `cuda-nvcc` provides an activation script (`~cuda-nvcc_activate.sh`) that automatically adds:
```bash
-I$CONDA_PREFIX/targets/x86_64-linux/include
-L$CONDA_PREFIX/targets/x86_64-linux/lib
```

---

### How this affects your `pixi.toml` and `build.rs`

Notice what you currently have in `pixi.toml`:

```toml
[activation.env]
CUDA_TOOLKIT_ROOT_DIR = "$CONDA_PREFIX"
CUDA_PATH = "$CONDA_PREFIX"
```

The Rust build script (`ct2rs/build.rs`) checks:
```rust
path.join("include").join("cuda.h").is_file()
```
Because `$CONDA_PREFIX/include/cuda.h` does not exist (it is at `$CONDA_PREFIX/targets/x86_64-linux/include/cuda.h`), `ct2rs` fails to detect CUDA under `$CONDA_PREFIX`.

To fix this, update your `pixi.toml` activation environment to:

```toml
[activation.env]
CFLAGS   = "$CFLAGS -pthread"
CXXFLAGS = "$CXXFLAGS -pthread"
CUDA_ARCH_LIST = "Auto"
CUDA_TOOLKIT_ROOT_DIR = "$CONDA_PREFIX/targets/x86_64-linux"
CUDA_PATH = "$CONDA_PREFIX/targets/x86_64-linux"
```
