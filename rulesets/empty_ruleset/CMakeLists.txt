
if(WITH_CUDA)
  set(CUDA_FILES empty_ruleset.cu empty_ruleset.cuh)
else()
  set(CUDA_FILES "")
endif()

add_library(empty_ruleset STATIC
  empty_ruleset.cpp
  empty_ruleset.h
  ${CUDA_FILES}
)

