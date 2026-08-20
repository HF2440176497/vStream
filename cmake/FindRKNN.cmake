# - Try to find RKNN runtime (RKNPU2, librknnrt.so)
#
# Input:
#   RKNN_ROOT_DIR  RKNPU2 根目录（默认: <仓库>/3rdparty/rknpu2）
#
# Output:
#   RKNN_FOUND
#   RKNN_INCLUDE_DIRS
#   RKNN_LIBRARIES

include(FindPackageHandleStandardArgs)

if(NOT RKNN_ROOT_DIR)
  if(DEFINED VSTREAM_ROOT_PATH)
    set(RKNN_ROOT_DIR "${VSTREAM_ROOT_PATH}/3rdparty/rknpu2")
  else()
    set(RKNN_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rknpu2")
  endif()
endif()
set(RKNN_ROOT_DIR "${RKNN_ROOT_DIR}" CACHE PATH "RKNPU2 runtime root (rknn_api.h + librknnrt.so)")

# 交叉编译下 CMAKE_FIND_ROOT_PATH_MODE_* = ONLY 会屏蔽仓库内路径（3rdparty 不在sysroot 内）
# 临时切换为 BOTH 查找，完成后恢复，避免污染其它查找
set(_rknn_saved_mode_lib ${CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
set(_rknn_saved_mode_inc ${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

find_path(RKNN_INCLUDE_DIR rknn_api.h
  PATHS ${RKNN_ROOT_DIR}/include
  NO_DEFAULT_PATH)

find_library(RKNN_LIBRARY
  NAMES rknnrt librknnrt.so
  PATHS ${RKNN_ROOT_DIR}/lib
  NO_DEFAULT_PATH)

set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${_rknn_saved_mode_lib})
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${_rknn_saved_mode_inc})

find_package_handle_standard_args(RKNN DEFAULT_MSG RKNN_LIBRARY RKNN_INCLUDE_DIR)

if(RKNN_FOUND)
  set(RKNN_INCLUDE_DIRS ${RKNN_INCLUDE_DIR})
  set(RKNN_LIBRARIES ${RKNN_LIBRARY})
  message(STATUS "Found RKNN runtime (include: ${RKNN_INCLUDE_DIRS}, library: ${RKNN_LIBRARIES})")
  mark_as_advanced(RKNN_INCLUDE_DIR RKNN_LIBRARY RKNN_ROOT_DIR)
endif()
