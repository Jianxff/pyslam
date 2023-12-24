# ----- Find dependencies -----

if(USE_STACK_TRACE_LOGGER)
    find_package(Glog REQUIRED)
    include_directories(${GLOG_INCLUDE_DIR})
    message(STATUS "Stack trace logger: ENABLED")
else()
    message(STATUS "Stack trace logger: DISABLED")
endif()


# ----- Show dialog -----

if(USE_PANGOLIN_VIEWER)
    message(STATUS "Viewer for examples: PangolinViewer")
elseif(USE_SOCKET_PUBLISHER)
    message(STATUS "Viewer for examples: SocketPublisher")
else()
    message(STATUS "Viewer for examples: None")
endif()

# ----- Build example executables -----
set(EXECUTABLE_TARGETS "")

add_executable(run_image_localization run_image_localization.cc util/image_util.cc)
list(APPEND EXECUTABLE_TARGETS run_image_localization)

# FW: Re-localization using point and line cloud
add_executable(run_image_localization_point_line run_image_localization_point_line.cc util/image_util.cc)
list(APPEND EXECUTABLE_TARGETS run_image_localization_point_line)

add_executable(run_euroc_slam run_euroc_slam.cc util/euroc_util.cc)
list(APPEND EXECUTABLE_TARGETS run_euroc_slam)

add_executable(run_euroc_slam_with_line run_euroc_slam_with_line.cc util/euroc_util.cc)
list(APPEND EXECUTABLE_TARGETS run_euroc_slam_with_line)


foreach(EXECUTABLE_TARGET IN LISTS EXECUTABLE_TARGETS)

    # PangolinViewer is used on a priority basis
    if(USE_PANGOLIN_VIEWER)
        # Set macro flag
        target_compile_definitions(${EXECUTABLE_TARGET} PRIVATE USE_PANGOLIN_VIEWER)

        # Link viewer
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE pangolin_viewer)
    elseif(USE_SOCKET_PUBLISHER)
        # Set macro flag
        target_compile_definitions(${EXECUTABLE_TARGET} PRIVATE USE_SOCKET_PUBLISHER)
        # Link viewer
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE socket_publisher)
    endif()

    # Setup stack trace logger
    if(USE_STACK_TRACE_LOGGER)
        target_compile_definitions(${EXECUTABLE_TARGET} PRIVATE USE_STACK_TRACE_LOGGER)
        target_link_libraries(${EXECUTABLE_TARGET} PRIVATE ${GLOG_LIBRARIES})
    endif()

    # Link PLPSLAM
    target_link_libraries(${EXECUTABLE_TARGET}
        PRIVATE
        ${PROJECT_NAME}
    )

    # include popl and spdlog headers
    target_include_directories(${EXECUTABLE_TARGET}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/3rd/popl/include
        ${PROJECT_SOURCE_DIR}/3rd/spdlog/include)
endforeach()
