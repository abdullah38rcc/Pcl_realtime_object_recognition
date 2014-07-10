
cmake_minimum_required(VERSION 2.8)
Project (pcl_object_recognition CXX)
add_executable (pcl_object_recognition pcl_object_recognition.cpp )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE Release)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -H")

find_path(OPENNI_INCLUDE_DIR NAMES OpenNI.h)
include_directories(${OPENNI_INCLUDE_DIR})
find_library(OPENNI_LIBRARY NAMES OpenNI2 libOpenNI2 )

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_path(PCLOPENNI2_INCLUDE_DIR openni2pcl.hpp)
include_directories(${PCLOPENNI2_INCLUDE_DIR} )


target_link_libraries( pcl_object_recognition  ${OPENNI_LIBRARY} ${PCL_LIBRARIES} )
