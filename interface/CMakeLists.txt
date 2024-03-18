##########################
# pybind interface
##########################
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)
message(STATUS "Found python ${PYTHON_VERSION_STRING} at: ${PYTHON_EXECUTABLE}")
message(STATUS "Python includes: ${PYTHON_INCLUDE_DIRS}")

pybind11_add_module(pysfm
    ${CMAKE_CURRENT_SOURCE_DIR}/interface_mvs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene_mvs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene_mvs.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/format_colmap.cc
    ${MAP_PB_SOURCE}
)

target_include_directories(pysfm
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/3rd/spdlog/include/
    ${PROJECT_SOURCE_DIR}/3rd/pybind11/include/
    ${PYTHON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(pysfm
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
  PRIVATE
    ${PROJECT_NAME}
)

if(BUILD_PANGOLIN_VIEWER)
  target_compile_definitions(pysfm PUBLIC WITH_PANGOLIN_VIEWER)
  target_link_libraries(pysfm PUBLIC pangolin_viewer ${Pangolin_LIBRARIES})
endif()

set_target_properties(pysfm PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${INTERFACE_OUTPUT_DIRECTORY})
set_target_properties(pysfm PROPERTIES OUTPUT_NAME "pysfm")
set_target_properties(pysfm PROPERTIES SUFFIX ".so")