##########################
# core library
##########################
add_subdirectory(core)

##########################
# socket publisher
##########################
if(USE_SOCKET_PUBLISHER)
  add_subdirectory(socket_publisher)
endif()

##########################
# pangolin viewer
##########################
if(USE_PANGOLIN_VIEWER)
  add_subdirectory(pangolin_viewer)
endif()
