# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/model.h
        ${CMAKE_CURRENT_SOURCE_DIR}/flann_neighborhood_graph.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GCRANSAC.h
        ${CMAKE_CURRENT_SOURCE_DIR}/grid_neighborhood_graph.h
        ${CMAKE_CURRENT_SOURCE_DIR}/neighborhood_graph.h
        ${CMAKE_CURRENT_SOURCE_DIR}/preemption_empty.h
        ${CMAKE_CURRENT_SOURCE_DIR}/preemption_sprt.h
        ${CMAKE_CURRENT_SOURCE_DIR}/progressive_napsac_sampler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/prosac_sampler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sampler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/scoring_function.h
        ${CMAKE_CURRENT_SOURCE_DIR}/settings.h
        ${CMAKE_CURRENT_SOURCE_DIR}/single_point_sampler.h
        ${CMAKE_CURRENT_SOURCE_DIR}/statistics.h
        ${CMAKE_CURRENT_SOURCE_DIR}/types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/uniform_random_generator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/uniform_sampler.h
)

add_subdirectory(pearl) # Graph-cut RANSAC
add_subdirectory(solver) # Graph-cut RANSAC
add_subdirectory(estimator) # Graph-cut RANSAC
