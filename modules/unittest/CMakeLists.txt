

if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
      CACHE INTERNAL "Top-level directory of this project")
endif()

include(${VSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
if(GTEST_FOUND)
  message(STATUS "Using installed GTest for modules unittest")
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  if(NOT has_built_gtest)
    add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/googletest)
    set(has_built_gtest TRUE)
  endif()
  include_directories(${VSTREAM_ROOT_PATH}/3rdparty/googletest/googletest/include)
endif()

# 必须包含框架头文件
include_directories(${VSTREAM_ROOT_PATH}/framework/core/include)
include_directories(${VSTREAM_ROOT_PATH}/framework/core/include/private)
include_directories(${VSTREAM_ROOT_PATH}/framework/core/include/util)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../util/include)  # modules/util 头文件

# unittest 头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(test_srcs "")
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# util 单元测试文件
file(GLOB test_util_srcs ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp)
list(APPEND test_srcs ${test_util_srcs})

# util 单元测试文件 [平台特定的单元测试文件]
if(VSTREAM_USE_CUDA)
  file(GLOB test_util_cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/util/cuda/*.cpp)
  list(APPEND test_srcs ${test_util_cuda_srcs})
  message(STATUS "Unittest: VSTREAM_USE_CUDA platform test sources added")
endif()

include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})
have_modules_target(${VSTREAM_ROOT_PATH})

if(HAVE_FRAMEWORK_TARGET AND HAVE_MODULES_TARGET)
  foreach(module ${module_list})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include)  # modules/$module_name/include
    file(GLOB_RECURSE test_${module}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${module}/*.cpp)
    list(APPEND test_srcs ${test_${module}_srcs})
  endforeach()
else()
  message(FATAL_ERROR "framework or modules not found")
endif()

add_executable(vstream_modules_test ${test_srcs})
add_dependencies(vstream_modules_test vstream_va vstream_core)

set(MODULES_TEST_DEPENDENCIES vstream_va vstream_core)

if(DEFINED MODULES_3RDPARTY_LIBS AND DEFINED MODULES_DEPENDENCIES)
  list(APPEND MODULES_TEST_DEPENDENCIES ${MODULES_3RDPARTY_LIBS})
  list(APPEND MODULES_TEST_DEPENDENCIES ${MODULES_DEPENDENCIES})
else()
  message(FATAL_ERROR "modules must define 3rd-party libraries and dependencies")
endif()

if(HAVE_FRAMEWORK_TARGET AND HAVE_MODULES_TARGET)
  if(DEFINED VSTREAM_LIB_PATH)
    set_target_properties(vstream_modules_test PROPERTIES
        BUILD_RPATH ${VSTREAM_LIB_PATH}
    )
  endif()
  if(GTEST_FOUND)
    target_link_libraries(vstream_modules_test ${MODULES_TEST_DEPENDENCIES} ${GTEST_LIBRARIES} glog gflags rt dl pthread)
  elseif(has_built_gtest)
    target_link_libraries(vstream_modules_test ${MODULES_TEST_DEPENDENCIES} gtest glog gflags rt dl pthread)
  endif()
endif()
