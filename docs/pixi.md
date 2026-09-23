Pixi Issues
-----------

# Setup

Move from full `cuda` (`coda-forge`) to minimal dependencies:

```toml
[dependencies]
cuda = ">=12.9,<13.0"
```

```toml
[dependencies]
rust = "*"

cuda-version = "12.9.*"
cudnn = "8.*"
cuda-nvcc = "*"
libcublas-dev = "*"
# cuda-cudart-dev = "*"
# libcurand-dev = "*"

# If WITH_TENSOR_PARALLEL
openmpi = "*"
nccl = "*"

compilers = ">=1.8.0"
cmake = ">=4.2.3,<5"
ninja = ">=1.13.2,<2"
# pkg-config = ">=0.29" <--- Why?

sccache = ">=0.17.0,<0.18" #TODO we must configurate sccache ASAP!
```

# NCCL > tensor-parallel
[NCCL](https://developer.nvidia.com/nccl)

The `CTranslate2` requires `NCCL` and `MPI` to build (line 500-502).

```CmakeLists.txt
if (WITH_TENSOR_PARALLEL)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
```
