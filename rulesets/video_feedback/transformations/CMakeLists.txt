
add_library(transformations STATIC
    #blend.cu
    #blend.h
    #brightness.cu
    #brightness.h
    #noise.cu
    #noise.h
    #rotation.cu
    #rotation.h
  transformation.cu
  transformation.h
  zoom.cu
  zoom.h
)

target_compile_options(transformations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -dc >)

target_link_libraries(transformations PRIVATE curand)

