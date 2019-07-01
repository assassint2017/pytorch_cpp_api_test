cmake_minimum_required(VERSION 3.0)
project(Torch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option (Debug "build with Debug otherwise build with Realease" OFF)

if (Debug)
    set(Torch_DIR "E:/Toolkits/libtorch/Debug/share/cmake/Torch")  # build with Debug
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(Torch_DIR "E:/Toolkits/libtorch/Realease/share/cmake/Torch")  # build with Realease
    set(CMAKE_BUILD_TYPE "Release")
endif (Debug)

find_package(Torch REQUIRED)
include_directories( ${TORCH_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(torch_cpp_test main.cpp)
target_link_libraries(torch_cpp_test ${TORCH_LIBRARIES} ${OpenCV_LIBS})