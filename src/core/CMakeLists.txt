add_library(${PROJECT_NAME} 
    SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/type.h
        ${CMAKE_CURRENT_SOURCE_DIR}/config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/system.h
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_module.h
        ${CMAKE_CURRENT_SOURCE_DIR}/mapping_module.h
        ${CMAKE_CURRENT_SOURCE_DIR}/global_optimization_module.h
        
        ${CMAKE_CURRENT_SOURCE_DIR}/config.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/system.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tracking_module.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapping_module.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/global_optimization_module.cc
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src/
        ${PROJECT_SOURCE_DIR}/3rd/json/include/
        ${PROJECT_SOURCE_DIR}/3rd/spdlog/include/
        ${SUITESPARSE_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC
        Threads::Threads
        yaml-cpp
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${SUITESPARSE_LIBRARIES}
        ${fbow_LIBS}
        g2o::core
        g2o::stuff
        g2o::types_sba
        g2o::types_sim3
        g2o::solver_dense
        g2o::solver_eigen
        g2o::solver_csparse
        g2o::csparse_extension
)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "slamcore")
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")

##########################
# sse3 optimize
##########################
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
target_compile_definitions(${PROJECT_NAME} PUBLIC USE_SSE_ORB)

##########################
# openmp
##########################
# find_package(OpenMP)
# if(OpenMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENMP)
# endif()

add_subdirectory(camera)
add_subdirectory(data)
add_subdirectory(feature)
add_subdirectory(initialize)
add_subdirectory(io)
add_subdirectory(match)
add_subdirectory(module)
add_subdirectory(optimize)
add_subdirectory(publish)
add_subdirectory(solve)
add_subdirectory(util)