##########################
# core library
##########################
add_subdirectory(core)

##########################
# socket publisher
##########################
if(BUILD_SOCKET_PUBLISHER)
  message(STATUS "Build with socket publisher")
  add_subdirectory(socket_publisher)
endif()

##########################
# pangolin viewer
##########################
if(BUILD_PANGOLIN_VIEWER)
  message(STATUS "Build with pangolin viewer")
  add_subdirectory(pangolin_viewer)
endif()
