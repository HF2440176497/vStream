

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_INCLUDE "")
set(VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_LIBS "")
set(VSTREAM_FRAMEWORK_UNITTEST_DEPENDENCIES "")

if(DEFINED VSTREAM_HAS_BUILT_GTEST AND VSTREAM_HAS_BUILT_GTEST)
  if(TARGET gtest::gtest)
    list(APPEND VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_LIBS gtest::gtest)
    message(STATUS "Using built GTest for framework unittest (target: gtest::gtest)")
  elseif(TARGET GTest::gtest)
    list(APPEND VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_LIBS GTest::gtest)
    message(STATUS "Using built GTest for framework unittest (target: GTest::gtest)")
  elseif(TARGET gtest)
    list(APPEND VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_LIBS gtest)
    message(STATUS "Using built GTest for framework unittest (target: gtest)")
  else()
    message(FATAL_ERROR "gtest target not found. Please ensure googletest is built")
  endif()
else()
  include(${VSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
  if(GTEST_FOUND)
    message(STATUS "Using installed GTest for framework unittest")
    list(APPEND VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_INCLUDE ${GTEST_INCLUDE_DIRS})
    list(APPEND VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_LIBS ${GTEST_LIBRARIES})
  else()
    message(FATAL_ERROR "GTest not found! Please enable VSTREAM_BUILD_TESTS to build googletest or install GTest")
  endif()
endif()

# ---[ source: framework unittest
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core core_test_srcs)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})

if (HAVE_FRAMEWORK_TARGET)
  list(APPEND VSTREAM_FRAMEWORK_UNITTEST_DEPENDENCIES vstream_core)
  add_executable(vstream_core_test ${core_test_srcs})
  target_include_directories(vstream_core_test 
   PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE}
    ${VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_INCLUDE}
  )
  target_link_libraries(vstream_core_test 
   PUBLIC
    ${VSTREAM_FRAMEWORK_UNITTEST_DEPENDENCIES}
   PRIVATE
    ${VSTREAM_FRAMEWORK_3RDPARTY_LIBS}
    ${VSTREAM_FRAMEWORK_UNITTEST_3RDPARTY_LIBS}
  )

  if(DEFINED VSTREAM_LIB_PATH)
    set_target_properties(vstream_core_test PROPERTIES
      BUILD_RPATH ${VSTREAM_LIB_PATH}
    )
  endif()
endif()

