# rulesets/ants

add_library(lifelike STATIC
  lifelike.cpp
  lifelike.h
)

target_compile_features(lifelike PUBLIC cxx_std_17)

