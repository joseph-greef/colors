# Root CMakeLists

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(colors LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  .
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${SDL2_INCLUDE_DIRS}
  cuda_kernels
  gifenc
  moviemaker-cpp/include
  rulesets
  rulesets/include
)

add_definitions(-DUSE_GPU)

add_subdirectory(rulesets)

add_subdirectory(moviemaker-cpp)

add_library(gifenc STATIC gifenc/gifenc.c gifenc/gifenc.h)

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

# Add rulesets
target_link_libraries(colors PRIVATE
  ants
  hodge
  lifelike
)

# Add general purpose libraries
target_link_libraries(colors PRIVATE
  curand
  gifenc
  moviemaker-cpp
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

