# SparseSpikes

Repository to implement cuda kernels for sparse spikes vectors for matrix multiply. This includes the kernels for forward and gradient calculation using surrogate gradients (forward and backward mode) for:
1. sparse spike vector times dense matrix matmul
2. sparse spike vector generaten from states and sthresholds
% 3. to_dense() or state_reset() function

The repository structure is as follows:
- `"lib/"` contains cuda kernels and neccessary c interface files
- `"jax_interface"` contains the python interface for jax and some test files 