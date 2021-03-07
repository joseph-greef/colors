# Root CMakeLists

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(SET CMP0091 NEW)
project(colors LANGUAGES C CXX CUDA)

set(CUDA_STATIC ON)
set(SDL2_STATIC ON)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenCV REQUIRED)
find_path(TCLAP_INCLUDE_DIRS "tclap/Arg.h")

include_directories(
  .
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${SDL2_INCLUDE_DIRS}
  ${TCLAP_INCLUDE_DIRS}
  cuda_kernels
  gifenc
  moviemaker-cpp/include
  rulesets
  rulesets/include
  rulesets/utilities
)

add_subdirectory(rulesets)

# Disable clip examples and tests
set(CLIP_EXAMPLES OFF CACHE BOOL "Compile clip examples")
set(CLIP_TESTS OFF CACHE BOOL "Compile clip tests")

# In case that you have ${PNG_LIBRARY} set to support copy/paste images on Linux
#set(CLIP_X11_PNG_LIBRARY "${PNG_LIBRARY}")

# Add clip subdirectory to compile the library
add_subdirectory(clip)

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

find_library(SDL_LIB_PATH "SDL2.lib")
get_filename_component(SDL_LIB_DIR ${SDL_LIB_PATH} DIRECTORY)
target_link_directories(colors PUBLIC ${SDL_LIB_DIR})

# Add rulesets
target_link_libraries(colors PRIVATE
  ants
  empty_ruleset
  hodge
  lifelike
  video_feedback
)

# Add general purpose libraries
target_link_libraries(colors PUBLIC
  clip
  cudart_static
  gifenc
  rulesets
  SDL2
  SDL2_image
  #utilities
  ${OpenCV_LIBS}
)

if(APPLE)
# We need to add the default path to the driver (libcuda.dylib) as an rpath,
# so that the static cuda runtime can find it at runtime.
set_property(TARGET colors PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

