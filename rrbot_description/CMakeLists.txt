cmake_minimum_required(VERSION 2.8.3)
project(rrbot_description)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(PROGRAMS
   nodes/publish_root_tf
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)