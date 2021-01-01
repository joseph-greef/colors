# rulesets/ants

add_library(lifelike STATIC
  lifelike.cpp
  lifelike.h
)

add_library(lifelike_cu STATIC
  lifelike.cu
  lifelike.cuh
)

target_link_libraries(lifelike PRIVATE lifelike_cu)

