
# 用于在 Windows 平台下方便地设置第三方库的路径
# 使用方法：在顶层 CMakeLists.txt 中 include 此文件
# 或者通过命令行参数：-DCMAKE_TOOLCHAIN_FILE=cmake/SetupLibraryPaths.cmake

if(UNIX)
    message(STATUS "=== Setting up library paths for Ubuntu ===")

    # CUDA 架构仅 x86+NVIDIA 平台需要；RK（aarch64）构建下跳过
    if(NOT VSTREAM_USE_ROCKCHIP)
        set(CMAKE_CUDA_ARCHITECTURES "86;89;120")
    endif()

    # FFmpeg：
    #   - RK 交叉编译默认指向 ffmpeg-rockchip 安装树（含 rkmpp 硬编解码器，
    #     由 toolchain 文件的 RK_FFMPEG_ROOT 提供）
    #   - 其余平台维持 /usr/local/ffmpeg
    if(NOT DEFINED FFMPEG_ROOT_DIR)
        if(VSTREAM_USE_ROCKCHIP AND RK_FFMPEG_ROOT)
            set(FFMPEG_ROOT_DIR "${RK_FFMPEG_ROOT}" CACHE PATH "Folder contains FFmpeg (ffmpeg-rockchip)")
            message(STATUS "FFMPEG_ROOT_DIR (rockchip): ${FFMPEG_ROOT_DIR}")
        else()
            set(FFMPEG_ROOT_DIR "/usr/local/ffmpeg" CACHE PATH "Folder contains FFmpeg")
            message(STATUS "FFMPEG_ROOT_DIR: ${FFMPEG_ROOT_DIR}")
        endif()
    else()
        message(STATUS "FFMPEG_ROOT_DIR (user defined): ${FFMPEG_ROOT_DIR}")
    endif()

    message(STATUS "=== Library paths setup complete ===")
else()
    message(STATUS "SetupLibraryPaths.cmake: Not on Linux, skipping automatic path setup")
endif()
