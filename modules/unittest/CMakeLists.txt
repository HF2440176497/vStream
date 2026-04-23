
if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
      CACHE INTERNAL "Top-level directory of this project")
endif()

set(VSTREAM_MODULES_UNITTEST_DEPENDENCIES "")
set(VSTREAM_MODULES_UNITTEST_3RDPARTY_INCLUDE "")
set(VSTREAM_MODULES_UNITTEST_3RDPARTY_LIBS "")

if(DEFINED VSTREAM_HAS_BUILT_GTEST AND VSTREAM_HAS_BUILT_GTEST)
  if(TARGET gtest::gtest)
    list(APPEND VSTREAM_MODULES_UNITTEST_3RDPARTY_LIBS gtest::gtest)
    message(STATUS "Using built GTest for modules unittest (target: gtest::gtest)")
  elseif(TARGET GTest::gtest)
    list(APPEND VSTREAM_MODULES_UNITTEST_3RDPARTY_LIBS GTest::gtest)
    message(STATUS "Using built GTest for modules unittest (target: GTest::gtest)")
  elseif(TARGET gtest)
    list(APPEND VSTREAM_MODULES_UNITTEST_3RDPARTY_LIBS gtest)
    message(STATUS "Using built GTest for modules unittest (target: gtest)")
  else()
    message(FATAL_ERROR "gtest target not found. Please ensure googletest is built")
  endif()
else()
  include(${VSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
  if(GTEST_FOUND)
    message(STATUS "Using installed GTest for modules unittest")
    list(APPEND VSTREAM_MODULES_UNITTEST_3RDPARTY_INCLUDE ${GTEST_INCLUDE_DIRS})
    list(APPEND VSTREAM_MODULES_UNITTEST_3RDPARTY_LIBS ${GTEST_LIBRARIES})
  else()
    message(FATAL_ERROR "GTest not found! Please enable VSTREAM_BUILD_TESTS to build googletest or install GTest")
  endif()
endif()

# include_directories(${VSTREAM_ROOT_PATH}/framework/core/include)
# include_directories(${VSTREAM_ROOT_PATH}/framework/core/include/private)
# include_directories(${VSTREAM_ROOT_PATH}/framework/core/include/util)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../util/include)  # vstream/modules/util/include

# # unittest 头文件
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
    # include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include)  # modules/$module_name/include
    file(GLOB_RECURSE test_${module}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${module}/*.cpp)
    list(APPEND test_srcs ${test_${module}_srcs})
  endforeach()
else()
  message(FATAL_ERROR "framework or modules not found")
endif()

add_executable(vstream_modules_test ${test_srcs})
list(APPEND VSTREAM_MODULES_UNITTEST_DEPENDENCIES vstream_va vstream_core)

if(HAVE_FRAMEWORK_TARGET AND HAVE_MODULES_TARGET)
  if(DEFINED VSTREAM_LIB_PATH)
    set_target_properties(vstream_modules_test PROPERTIES
      BUILD_RPATH ${VSTREAM_LIB_PATH}
    )
  endif()
  target_include_directories(vstream_modules_test 
   PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VSTREAM_MODULES_3RDPARTY_INCLUDE}
    ${VSTREAM_MODULES_UNITTEST_3RDPARTY_INCLUDE}
  )
  target_link_libraries(vstream_modules_test
   PUBLIC
    ${VSTREAM_MODULES_DEPENDENCIES}
    ${VSTREAM_MODULES_UNITTEST_DEPENDENCIES}
   PRIVATE
    ${VSTREAM_MODULES_3RDPARTY_LIBS}
    ${VSTREAM_MODULES_UNITTEST_3RDPARTY_LIBS}
  )
endif()
