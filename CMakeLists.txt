cmake_minimum_required(VERSION 3.16)
project(lycklig
    VERSION 1.0.0
    DESCRIPTION "Astronomical image processing pipeline for lucky imaging"
    LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TCLAP REQUIRED IMPORTED_TARGET tclap)
pkg_check_modules(MAGICK REQUIRED IMPORTED_TARGET Magick++)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenMP REQUIRED)

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

target_compile_options(lycklig PRIVATE -Wall)
target_compile_options(lycklig PRIVATE "$<$<CONFIG:Release>:-O3>")
target_compile_options(lycklig PRIVATE "$<$<CONFIG:RelWithDebInfo>:-O3>")

target_link_libraries(lycklig PRIVATE
  ${OpenCV_LIBS}
  PkgConfig::TCLAP
  PkgConfig::MAGICK
  Boost::filesystem
  Boost::system
  OpenMP::OpenMP_CXX
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
