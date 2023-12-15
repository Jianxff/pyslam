##########################
# pangolin
##########################
find_package(Pangolin REQUIRED)

add_library(pangolin_viewer
    SHARED
        viewer.h
        color_scheme.h
        viewer.cc
        color_scheme.cc
)

target_link_libraries(pangolin_viewer
    PUBLIC
        ${PROJECT_NAME}
        ${Pangolin_LIBRARIES}
        ${OpenCV_LIBS}
)
