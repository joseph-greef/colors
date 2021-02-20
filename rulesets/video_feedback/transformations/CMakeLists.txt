
add_library(transformations STATIC
  blend.cpp
  blend.h
  noise.cpp
  noise.h
  rotation.cpp
  rotation.h
  transformation.cpp
  transformation.h
  zoom.cpp
  zoom.h
)

target_compile_features(transformations PUBLIC cxx_std_17)
