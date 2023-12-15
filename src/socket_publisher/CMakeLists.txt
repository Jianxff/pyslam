# ----- Find dependencies -----

# socket.io-cpp-client
find_package(sioclient REQUIRED)

# Protobuf
find_package(Protobuf REQUIRED)

if(NOT PROTOBUF_PROTOC_EXECUTABLE)
    message(FATAL_ERROR "Could not find protoc executable (PROTOBUF_PROTOC_EXECUTABLE)")
endif()

message(STATUS "Found protoc executable: ${PROTOBUF_PROTOC_EXECUTABLE}")

# ----- Protobuf transpile -----
set(PROTOBUF_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(MAP_PB_SOURCE MAP_PB_HEADER protobuf/map_segment.proto)
set_source_files_properties(${MAP_PB_HEADER} ${MAP_PB_SOURCE}
    COMPILE_FLAGS -Wno-unused-variable)

# ----- Configure SocketPublisher library -----
add_library(socket_publisher
    SHARED
    data_serializer.h
    publisher.h
    socket_client.h

    data_serializer.cc
    publisher.cc
    socket_client.cc
    ${MAP_PB_SOURCE})

target_include_directories(socket_publisher
    PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(socket_publisher
    PUBLIC
    ${PROJECT_NAME}
    opencv_imgcodecs
    sioclient
    ${PROTOBUF_LIBRARIES})