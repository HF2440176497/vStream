cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VSTREAM_PYTHON_3RDPARTY_INCLUDE "")
set(VSTREAM_PYTHON_3RDPARTY_LIBS "")
set(VSTREAM_PYTHON_DEPENDENCIES "")

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for python api")
  list(APPEND VSTREAM_PYTHON_3RDPARTY_INCLUDE ${GFLAGS_INCLUDE_DIRS})
  list(APPEND VSTREAM_PYTHON_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for python api")
  list(APPEND VSTREAM_PYTHON_3RDPARTY_INCLUDE ${GLOG_INCLUDE_DIRS})
  list(APPEND VSTREAM_PYTHON_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ nlohmann/json
list(APPEND VSTREAM_PYTHON_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ framework and modules
include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})
have_modules_target(${VSTREAM_ROOT_PATH})
if(HAVE_FRAMEWORK_TARGET AND HAVE_MODULES_TARGET)

  # include_directories(${VSTREAM_ROOT_PATH}/framework/core/include)

  # include_directories(${VSTREAM_ROOT_PATH}/modules)
  # include_directories(${VSTREAM_ROOT_PATH}/modules/util/include)

  # modules parts include
  # list(APPEND python_api_module_list source inference osd)
  #   for(module ${python_api_module_list})
  #     include_directories(${VSTREAM_ROOT_PATH}/modules/${module}/include)
  #   endforeach()

  list(APPEND VSTREAM_PYTHON_DEPENDENCIES vstream_core vstream_va)
else()
  message(FATAL_ERROR "framework target not found")
endif()

# ---[ pybind11
if(DEFINED VSTREAM_HAS_BUILT_PYBIND11 AND VSTREAM_HAS_BUILT_PYBIND11)
  message(STATUS "Using pybind11 from 3rdparty build")
else()
  # 系统中如果安装，使用 find_package 查找
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(${VSTREAM_ROOT_PATH}/cmake/FindPybind11.cmake)
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR "Pybind11 not found. Install it or enable VSTREAM_BUILD_PYBIND11")
  endif()
endif()

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  list(APPEND VSTREAM_PYTHON_3RDPARTY_INCLUDE ${OpenCV_INCLUDE_DIRS})
  list(APPEND VSTREAM_PYTHON_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ vstream_python srcs
set(vstream_pybind_srcs "")
file(GLOB vstream_pybind_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

pybind11_add_module(vstream_python
  ${vstream_pybind_srcs}
)

target_include_directories(vstream_python
 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include  # vstream_python is final object
  ${VSTREAM_PYTHON_3RDPARTY_INCLUDE}
)

target_link_libraries(vstream_python 
 PUBLIC
  ${VSTREAM_PYTHON_DEPENDENCIES}
 PRIVATE
  ${VSTREAM_PYTHON_3RDPARTY_LIBS}
)

set_target_properties(vstream_python PROPERTIES
  OUTPUT_NAME "vstream"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vstream
)

if(VSTREAM_BUILD_TESTS)
  add_subdirectory(unittest)
endif()
