# Root CMakeLists

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
cmake_policy(SET CMP0091 NEW)
project(colors LANGUAGES C CXX CUDA)

set(CUDA_STATIC ON)
set(SDL2_STATIC ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


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
  gifenc
  src
  src/rulesets
  src/rulesets/utilities
  src/rulesets/utilities/rainbows
)


# Disable clip examples and tests
set(CLIP_EXAMPLES OFF CACHE BOOL "Compile clip examples")
set(CLIP_TESTS OFF CACHE BOOL "Compile clip tests")

# In case that you have ${PNG_LIBRARY} set to support copy/paste images on Linux
#set(CLIP_X11_PNG_LIBRARY "${PNG_LIBRARY}")

# Add clip subdirectory to compile the library
add_subdirectory(clip)

add_library(gifenc STATIC gifenc/gifenc.c gifenc/gifenc.h)

add_subdirectory(src)

