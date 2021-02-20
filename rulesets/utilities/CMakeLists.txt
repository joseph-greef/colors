# utilities

add_library(utilities STATIC
    imgproc.cu
    imgproc.cuh
)

target_compile_options(utilities PRIVATE -dc)

