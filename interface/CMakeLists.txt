##########################
# protobuf generate
##########################
find_package(Protobuf REQUIRED)

set(PROTOBUF_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(MAP_PB_SOURCE MAP_PB_HEADER protobuf/pointcloud.proto)
set_source_files_properties(${MAP_PB_HEADER} ${MAP_PB_SOURCE}
    COMPILE_FLAGS -Wno-unused-variable)

##########################
# pybind interface
##########################
pybind11_add_module(pysfm
    ${CMAKE_CURRENT_SOURCE_DIR}/interface_mvs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene_mvs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pcl_serializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scene_mvs.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/pcl_serializer.cc
    ${MAP_PB_SOURCE}
)

target_include_directories(pysfm
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/3rd/spdlog/include/
    ${PROJECT_SOURCE_DIR}/3rd/pybind11/include/
    ${PYTHON_INCLUDE_DIRECTORY}
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(pysfm
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PROTOBUF_LIBRARIES}
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