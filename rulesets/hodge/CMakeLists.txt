# rulesets/ants

add_library(hodge STATIC
  hodge.cpp
  hodge.h
)

add_library(hodge_cu STATIC
  hodge.cu
  hodge.cuh
)

target_link_libraries(hodge PRIVATE hodge_cu)

