
# 用于在 Windows 平台下方便地设置第三方库的路径
# 使用方法：在顶层 CMakeLists.txt 中 include 此文件
# 或者通过命令行参数：-DCMAKE_TOOLCHAIN_FILE=cmake/SetupLibraryPaths.cmake

if(UNIX)
    message(STATUS "=== Setting up library paths for Ubuntu ===")

    set(CMAKE_CUDA_ARCHITECTURES "86;89;120")

    # FFmpeg
    if(NOT DEFINED FFMPEG_ROOT_DIR)
        set(FFMPEG_ROOT_DIR "/usr/local/ffmpeg" CACHE PATH "Folder contains FFmpeg")
        message(STATUS "FFMPEG_ROOT_DIR: ${FFMPEG_ROOT_DIR}")
    else()
        message(STATUS "FFMPEG_ROOT_DIR (user defined): ${FFMPEG_ROOT_DIR}")
    endif()

    message(STATUS "=== Library paths setup complete ===")
else()
    message(STATUS "SetupLibraryPaths.cmake: Not on Linux, skipping automatic path setup")
endif()
