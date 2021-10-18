project(lycklig)
cmake_minimum_required(VERSION 3.0)

find_package(OpenCV REQUIRED)
include(FindPkgConfig)
pkg_check_modules(PKGCONFS REQUIRED tclap Magick++)
find_package(Boost REQUIRED COMPONENTS filesystem system)

string(REPLACE ";" " " PKGCONFS_CFLAGS "${PKGCONFS_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp ${PKGCONFS_CFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")

add_executable(lycklig
  src/cookedtemplate.cpp
  src/globalregistrator.cpp
  src/imageops.cpp
  src/imagepatch.cpp
  src/dedistort.cpp
  src/main.cpp
  src/rbfwarper.cpp
  src/registrationcontext.cpp
  src/registrationparams.cpp
)

target_link_libraries(lycklig
  ${OpenCV_LIBS}
  ${PKGCONFS_LDFLAGS}
  ${Boost_LIBRARIES}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/kinky/kinky.py
  ${CMAKE_CURRENT_BINARY_DIR}/kinky
  @ONLY)

install(TARGETS lycklig
  RUNTIME
  DESTINATION bin)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kinky
  DESTINATION bin)
install(FILES kinky/kinky.desktop
  DESTINATION share/applications/)
install(FILES kinky/kinky.svgz
  RENAME application-x-kinky.svgz
  DESTINATION share/icons/)
