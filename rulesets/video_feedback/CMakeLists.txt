
add_subdirectory(transformations)

add_library(video_feedback STATIC
  video_feedback.cpp
  video_feedback.h
)

target_link_libraries(video_feedback PRIVATE transformations)

