
add_library(ants STATIC
  ant.h
  ants.cpp
  ants.h
  colony.cpp
  colony.h
  food.h
)

target_compile_features(ants PUBLIC cxx_std_17)
