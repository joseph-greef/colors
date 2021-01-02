# rulesets/ants

if(WITH_CUDA)
  set(CUDA_FILES lifelike.cu lifelike.cuh)
else()
  set(CUDA_FILES "")
endif()

add_library(lifelike STATIC
  lifelike.cpp
  lifelike.h
  ${CUDA_FILES}
)

