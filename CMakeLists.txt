cmake_minimum_required(VERSION 3.8)
project(PLSLAM LANGUAGES CXX C)

cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0079 NEW)

##########################
# cmake standard
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

##########################
# cmake warnings
##########################
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -Wno-ignored-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -Wno-ignored-attributes")


##########################
# cmake optimize
##########################
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Og -ffast-math -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Og -ffast-math -mtune=native")


##########################
# device optimize
##########################
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -march=native")


##########################
# ccache
##########################
find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_EXE}")
    message(STATUS "ccache: Enabled")
endif()

##########################
# sanitizer
##########################
# set(USE_SANITIZER OFF BOOL)
# if(USE_SANITIZER)
#     add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
#     set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
#     set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
# endif()


##########################
# set 3rd directory
##########################
set(fbow_DIR ${PROJECT_SOURCE_DIR}/3rd/FBoW/install/share/cmake/fbow)


##########################
# find package
##########################
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fbow REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED 
  COMPONENTS
    g2o::core
    g2o::stuff
    g2o::types_sba
    g2o::types_sim3
    g2o::solver_dense
    g2o::solver_eigen
    g2o::solver_csparse
    g2o::csparse_extension
)
find_package(CXSparse)
if(NOT CXSPARSE_FOUND)
  find_package(SuiteSparse REQUIRED)
endif()


##########################
# build library
##########################
# set output
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(src)

##########################
# extend library
##########################
set(BUILD_SOCKET_PUBLISHER OFF CACHE BOOL "Enable Socket Publisher")
set(BUILD_PANGOLIN_VIEWER ON CACHE BOOL "Enable Pangolin Publisher")


##########################
# pybind11 interface
##########################
set(BUILD_INTERFACE ON CACHE BOOL "Build Pybind Interface")
if(BUILD_INTERFACE)
  message(STATUS "Build Pybind Interface")

  set(INTERFACE_OUTPUT_DIRECTORY ${INTERFACE_OUTPUT})
  if(NOT INTERFACE_OUTPUT_DIRECTORY)
    set(INTERFACE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  endif()
  
  message(STATUS "Interface output directory: ${INTERFACE_OUTPUT_DIRECTORY}")

  add_subdirectory(${PROJECT_SOURCE_DIR}/3rd/pybind11)
  add_subdirectory(interface)
endif()

##########################
# examples
##########################
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
if(BUILD_EXAMPLES)
  message(STATUS "Build examples")

  set(USE_STACK_TRACE_LOGGER OFF CACHE BOOL "Enable automatic stack trace logger of google-glog")
  set(USE_PANGOLIN_VIEWER OFF CACHE BOOL "Use Pangolin Viewer")
  set(USE_SOCKET_PUBLISHER OFF CACHE BOOL "Use Socket Publisher")
  add_subdirectory(examples)
endif()
