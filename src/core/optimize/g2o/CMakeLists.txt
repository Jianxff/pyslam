# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex.h
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_container.h
        ${CMAKE_CURRENT_SOURCE_DIR}/line3d.h
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_line3d.h
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_container_line3d.h
        
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_container.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/line3d.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_line3d.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/landmark_vertex_container_line3d.cc
)

# Append subdirectory
add_subdirectory(se3)
add_subdirectory(sim3)

# add_subdirectory(extended) # for additional landmark plane