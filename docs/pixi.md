Pixi Issues
-----------

# NCCL > tensor-parallel
[NCCL](https://developer.nvidia.com/nccl)

The `CTranslate2` requires `NCCL` and `MPI` to build (line 500-502).

```CmakeLists.txt
if (WITH_TENSOR_PARALLEL)
  find_package(MPI REQUIRED)
  find_package(NCCL REQUIRED)
```
