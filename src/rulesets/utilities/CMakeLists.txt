
add_library(utilities STATIC
  initializer.cpp
  ruleset.cpp
  initializer.h
  ruleset.h
)

target_compile_features(utilities PUBLIC cxx_std_17)

