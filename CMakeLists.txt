cmake_minimum_required(VERSION 3.10)
project(SeamCarving)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SeamCarving src/main.cpp src/seam_carver.cpp)
target_link_libraries(SeamCarving ${OpenCV_LIBS})

# It is good practice to set the output directory for executables
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 