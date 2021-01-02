# rulesets/ants

if(WITH_CUDA)
  set(CUDA_FILES hodge.cu hodge.cuh)
else()
  set(CUDA_FILES "")
endif()

add_library(hodge STATIC
  hodge.cpp
  hodge.h
  ${CUDA_FILES}
)

