# cuda_kernels

add_library(cuda_kernels STATIC
  hodge.cu
  hodge.cuh
  lifelike.cu
  lifelike.cuh
)

target_compile_features(cuda_kernels PUBLIC cxx_std_11)

set_target_properties(cuda_kernels
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

