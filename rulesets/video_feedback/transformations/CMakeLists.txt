
add_library(transformations STATIC
  blend.cu
  blend.h
  noise.cu
  noise.h
  rotation.cu
  rotation.h
  transformation.cpp
  transformation.h
  zoom.cu
  zoom.h
)

target_compile_options(transformations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -dc >)

target_link_libraries(transformations PRIVATE utilities)
