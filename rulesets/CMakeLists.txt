# rulesets

add_subdirectory(ants)

add_library(rulesets STATIC hodge.cpp
                            hodge.h
                            initializer.cpp
                            initializer.h
                            lifelike.cpp
                            lifelike.h
                            rainbows.cpp
                            rainbows.h
                            ruleset.cpp
                            ruleset.h
)

target_compile_features(rulesets PUBLIC cxx_std_17)

target_link_libraries(rulesets PRIVATE ants gifenc)

