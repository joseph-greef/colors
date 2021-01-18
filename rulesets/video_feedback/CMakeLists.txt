
if(WITH_CUDA)
  set(CUDA_FILES video_feedback.cu video_feedback.cuh)
else()
  set(CUDA_FILES "")
endif()

add_library(video_feedback STATIC
  video_feedback.cpp
  video_feedback.h
  ${CUDA_FILES}
)

