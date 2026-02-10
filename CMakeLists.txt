cmake_minimum_required(VERSION 4.0.2)
project(OpenCVExample)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_VERSION_MAJOR)
    list(GET GCC_VERSION_COMPONENTS 1 GCC_VERSION_MINOR)

    if(GCC_VERSION_MAJOR LESS 9)
        target_link_libraries(my_program stdc++fs)
    endif()
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})


include_directories(${CMAKE_SOURCE_DIR}/OpenCl/include)

add_executable(OpenCVExample 
    PRSVideoObrada.cpp
)

target_link_libraries(OpenCVExample ${OpenCV_LIBS} ${OpenCL_LIBRARIES})
target_link_libraries(OpenCVExample OpenCL::OpenCL)

