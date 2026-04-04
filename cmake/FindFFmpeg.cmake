# ==============================================
# Try to find FFmpeg libraries:
# - avcodec
# - avformat
# - avdevice
# - avutil
# - swscale
# - avfilter
#
# FFMPEG_FOUND - system has FFmpeg
# FFMPEG_INCLUDE_DIR - the FFmpeg inc directory
# FFMPEG_LIBRARIES - Link these to use FFmpeg
# ==============================================
# Notice: this original script is only for Linux.

# Support custom FFmpeg installation path
if(NOT DEFINED FFMPEG_ROOT_DIR)
    set(FFMPEG_ROOT_DIR "/usr/local/" CACHE PATH "Default FFmpeg root directory")
endif()

if (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
    # in cache already
    set(FFMPEG_FOUND TRUE)
else (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
    find_path(
            FFMPEG_AVCODEC_INCLUDE_DIR
            NAMES libavcodec/avcodec.h
            PATHS ${FFMPEG_ROOT_DIR}/include
            /usr/include/ffmpeg
            /usr/local/include
   	    /usr/include/x86_64-linux-gnu
            /usr/include/aarch64-linux-gnu
    )

    find_library(
            FFMPEG_LIBAVCODEC
            NAMES avcodec
            PATHS ${FFMPEG_ROOT_DIR}/lib
            /usr/lib64
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
    )
  
    find_library(
            FFMPEG_LIBAVFORMAT
            NAMES avformat
            PATHS ${FFMPEG_ROOT_DIR}/lib

            /usr/lib64
            /usr/local/lib
	    /usr/lib/x86_64-linux-gnu
    )
  
    find_library(
            FFMPEG_LIBSWRESAMPLE
            NAMES swresample
            PATHS ${FFMPEG_ROOT_DIR}/lib
            /usr/lib64
            /usr/local/lib
	    /usr/lib/x86_64-linux-gnu
    )
 
    find_library(
            FFMPEG_LIBAVUTIL
            NAMES avutil
            PATHS ${FFMPEG_ROOT_DIR}/lib
            /usr/lib64
            /usr/local/lib
	    /usr/lib/x86_64-linux-gnu
    )

    find_library(
            FFMPEG_LIBSWSCALE
            NAMES swscale
            PATHS ${FFMPEG_ROOT_DIR}/lib
            /usr/lib64
            /usr/local/lib
	    /usr/lib/x86_64-linux-gnu
    )

    find_library(
            FFMPEG_LIBAVFILTER
            NAMES avfilter
            PATHS ${FFMPEG_ROOT_DIR}/lib
            /usr/lib64
            /usr/local/lib
	    /usr/lib/x86_64-linux-gnu
    )
    find_library(
            FFMPEG_LIBAVDEVICE
            NAMES avdevice
            PATHS ${FFMPEG_ROOT_DIR}/lib
            /usr/lib64
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
    )
  if (NOT FFMPEG_LIBAVDEVICE)
        message(FATAL_ERROR "Not find FFmpeg LIBAVDEVICE ")
  endif ()
  if (FFMPEG_LIBAVCODEC AND FFMPEG_LIBAVFORMAT AND FFMPEG_LIBAVUTIL AND FFMPEG_LIBSWSCALE AND FFMPEG_LIBSWRESAMPLE AND FFMPEG_LIBAVDEVICE)
        set(FFMPEG_FOUND TRUE)
  endif ()

  if (FFMPEG_FOUND)
        set(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR})
        set(FFMPEG_LIBRARIES
                ${FFMPEG_LIBAVCODEC}
                ${FFMPEG_LIBAVFORMAT}
                ${FFMPEG_LIBAVUTIL}
                ${FFMPEG_LIBSWSCALE}
                ${FFMPEG_LIBSWRESAMPLE}
                ${FFMPEG_LIBAVDEVICE})
  else (FFMPEG_FOUND)
        message(FATAL_ERROR "Could not find FFmpeg libraries!")
  endif (FFMPEG_FOUND)

endif (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
