# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/estimator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/essential_estimator.h

        # ${CMAKE_CURRENT_SOURCE_DIR}/perspective_n_point_estimator.h
        # ${CMAKE_CURRENT_SOURCE_DIR}/rigid_transformation_estimator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/sample_consensus_estimator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/fundamental_estimator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/homography_estimator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/linear_model_estimator.h
)
