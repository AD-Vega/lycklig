project(lycklig)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(FindPkgConfig)
pkg_check_modules(PKGCONFS REQUIRED tclap Magick++)

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

set_prefixed(lycklig_SRCS src/ imageops.cpp main.cpp rbfwarper.cpp
             registrationparams.cpp)
add_executable(lycklig ${lycklig_SRCS})
target_link_libraries(lycklig ${OpenCV_LIBS} ${PKGCONFS_LDFLAGS})

install(TARGETS lycklig RUNTIME DESTINATION bin)
