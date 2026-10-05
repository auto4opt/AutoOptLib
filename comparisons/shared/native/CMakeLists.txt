cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(autooptlib_eofastga_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PARADISEO_ROOT "" CACHE PATH "Official ParadisEO source checkout")
set(PARADISEO_BUILD "" CACHE PATH "Official ParadisEO build directory")
set(IOH_ROOT "" CACHE PATH "Official IOHexperimenter source checkout")

foreach(required PARADISEO_ROOT PARADISEO_BUILD IOH_ROOT)
  if(NOT ${required})
    message(FATAL_ERROR "${required} must be provided")
  endif()
endforeach()

find_path(PARADISEO_EO_INCLUDE eo PATHS "${PARADISEO_ROOT}/eo/src" NO_DEFAULT_PATH)
find_path(IOH_INCLUDE ioh.hpp PATHS "${IOH_ROOT}/include" NO_DEFAULT_PATH)
find_path(IOH_JSON_INCLUDE nlohmann/json.hpp
  PATHS "${IOH_ROOT}/external/json/include" NO_DEFAULT_PATH)
find_path(IOH_FMT_INCLUDE fmt/format.h
  PATHS "${IOH_ROOT}/external/fmt/include" NO_DEFAULT_PATH)

foreach(required PARADISEO_EO_INCLUDE IOH_INCLUDE IOH_JSON_INCLUDE IOH_FMT_INCLUDE)
  if(NOT ${required})
    message(FATAL_ERROR "Could not resolve ${required}")
  endif()
endforeach()

foreach(library eo eoutils ga es)
  find_library(PARADISEO_${library}_LIBRARY ${library}
    PATHS "${PARADISEO_BUILD}/lib" NO_DEFAULT_PATH REQUIRED)
  list(APPEND PARADISEO_LIBRARIES "${PARADISEO_${library}_LIBRARY}")
endforeach()

add_executable(autooptlib-fastga-runner fastga_runner.cpp)
target_include_directories(autooptlib-fastga-runner PRIVATE
  "${PARADISEO_ROOT}"
  "${PARADISEO_EO_INCLUDE}"
  "${IOH_INCLUDE}"
  "${IOH_JSON_INCLUDE}"
  "${IOH_FMT_INCLUDE}"
)
target_compile_definitions(autooptlib-fastga-runner PRIVATE FMT_HEADER_ONLY=1)
target_compile_options(autooptlib-fastga-runner PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
)
target_link_libraries(autooptlib-fastga-runner PRIVATE ${PARADISEO_LIBRARIES})
