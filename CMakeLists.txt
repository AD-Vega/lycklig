project(lycklig)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(FindPkgConfig)
pkg_check_modules(PKGCONFS REQUIRED tclap Magick++)
find_package(Boost REQUIRED COMPONENTS filesystem system)

macro(set_prefixed varname string)
  set(${varname})
  foreach(name ${ARGN})
    set( ${varname} ${${varname}} ${string}${name} )
  endforeach(name)
endmacro(set_prefixed)

string(REPLACE ";" " " PKGCONFS_CFLAGS "${PKGCONFS_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -fopenmp ${PKGCONFS_CFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")

set_prefixed(lycklig_SRCS src/ cookedtemplate.cpp globalregistrator.cpp
             imageops.cpp imagepatch.cpp lucky.cpp main.cpp rbfwarper.cpp
             registrationcontext.cpp registrationparams.cpp)
add_executable(lycklig ${lycklig_SRCS})
target_link_libraries(lycklig ${OpenCV_LIBS} ${PKGCONFS_LDFLAGS} ${Boost_LIBRARIES})

install(TARGETS lycklig RUNTIME DESTINATION bin)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/kinky/kinky.py
               ${CMAKE_CURRENT_BINARY_DIR}/kinky
               @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/kinky DESTINATION bin)
install(FILES kinky/kinky.desktop
  DESTINATION share/applications/)
install(FILES kinky/kinky.svgz
  RENAME application-x-kinky.svgz
  DESTINATION share/icons/)
