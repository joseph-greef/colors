
add_subdirectory(rainbows)

add_library(utilities STATIC
  initializer.cpp
  ruleset.cpp
  initializer.h
  ruleset.h
)

target_link_libraries(utilities PUBLIC
  rainbows
)

target_compile_features(utilities PUBLIC cxx_std_17)

