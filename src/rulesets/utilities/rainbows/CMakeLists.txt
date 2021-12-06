
add_library(rainbows STATIC
  rainbows.cpp
  rainbows.h
  rainbows.cu
  rainbows.cuh
)

target_link_libraries(rainbows PUBLIC gifenc)

