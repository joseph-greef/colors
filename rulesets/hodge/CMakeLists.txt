# rulesets/ants

add_library(hodge STATIC
  hodge.cpp
  hodge.h
)

target_compile_features(hodge PUBLIC cxx_std_17)

