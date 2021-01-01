# rulesets

add_subdirectory(ants)
add_subdirectory(hodge)
add_subdirectory(lifelike)

add_library(rulesets STATIC
  initializer.cpp
  rainbows.cpp
  ruleset.cpp
  include/initializer.h
  include/rainbows.h
  include/ruleset.h
)

target_compile_features(rulesets PUBLIC cxx_std_17)

target_link_libraries(rulesets PUBLIC gifenc)

