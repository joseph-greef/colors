# Root CMakeLists

add_subdirectory(rulesets)

add_executable(colors
  game.cpp
  game.h
  input_manager.cpp
  input_manager.h
  main.cpp
)

target_compile_features(colors PUBLIC cxx_std_17)

set_target_properties(colors
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

find_library(SDL_LIB_PATH "SDL2.lib")
get_filename_component(SDL_LIB_DIR ${SDL_LIB_PATH} DIRECTORY)
target_link_directories(colors PUBLIC ${SDL_LIB_DIR})

# Add rulesets
target_link_libraries(colors PRIVATE
  ants
  empty_ruleset
  hodge
  lifelike
)

# Add general purpose libraries
target_link_libraries(colors PRIVATE
  clip
  cudart_static
  gifenc
  rulesets
  SDL2
  SDL2_image
  ${OpenCV_LIBS}
)

if(APPLE)
# We need to add the default path to the driver (libcuda.dylib) as an rpath,
# so that the static cuda runtime can find it at runtime.
set_property(TARGET colors PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

