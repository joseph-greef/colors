
add_library(utilities STATIC
  initializer.cpp
  rainbows.cpp
  ruleset.cpp
  initializer.h
  rainbows.h
  ruleset.h
)

target_compile_features(utilities PUBLIC cxx_std_17)

target_link_libraries(utilities PUBLIC gifenc)
