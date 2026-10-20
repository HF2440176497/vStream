# =============================================================================
# cmake/SetupLibraryPaths_ubuntu.cmake
# 第三方库路径默认值
#
# 平台默认值由 cmake/platforms/<id>.cmake 提供（VSTREAM_PLATFORM_FFMPEG_ROOT /
# VSTREAM_PLATFORM_OPENCV_ROOT），本文件只负责：
#   1. 用户显式 -D 优先
#   2. 未指定时用平台默认值
#   3. 平台无默认值时用通用兜底值
# =============================================================================

if(NOT UNIX)
  message(STATUS "SetupLibraryPaths: not on Linux, skipping automatic path setup")
  return()
endif()

# ---- FFmpeg ----
if(NOT DEFINED FFMPEG_ROOT_DIR)
  if(VSTREAM_PLATFORM_FFMPEG_ROOT)
    set(FFMPEG_ROOT_DIR "${VSTREAM_PLATFORM_FFMPEG_ROOT}" CACHE PATH "Folder contains FFmpeg")
    message(STATUS "FFMPEG_ROOT_DIR (${VSTREAM_PLATFORM_ID}): ${FFMPEG_ROOT_DIR}")
  else()
    set(FFMPEG_ROOT_DIR "/usr/local/ffmpeg" CACHE PATH "Folder contains FFmpeg")
    message(STATUS "FFMPEG_ROOT_DIR: ${FFMPEG_ROOT_DIR}")
  endif()
else()
  message(STATUS "FFMPEG_ROOT_DIR (user defined): ${FFMPEG_ROOT_DIR}")
endif()

# ---- OpenCV ----
# 平台提供预编译树（如 SDK sysroot）时才设置；否则留给 FindOpenCV 走系统默认搜索
if(NOT DEFINED OPENCV_ROOT_DIR)
  if(VSTREAM_PLATFORM_OPENCV_ROOT)
    set(OPENCV_ROOT_DIR "${VSTREAM_PLATFORM_OPENCV_ROOT}" CACHE PATH "Folder contains OpenCV")
    message(STATUS "OPENCV_ROOT_DIR (${VSTREAM_PLATFORM_ID}): ${OPENCV_ROOT_DIR}")
  endif()
else()
  message(STATUS "OPENCV_ROOT_DIR (user defined): ${OPENCV_ROOT_DIR}")
endif()
