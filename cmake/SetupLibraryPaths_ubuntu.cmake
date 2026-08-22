
# 用于设置第三方库的路径
# 使用方法：在顶层 CMakeLists.txt 中 include 此文件
# 或者通过命令行参数：-DCMAKE_TOOLCHAIN_FILE=cmake/SetupLibraryPaths.cmake

if(UNIX)
    message(STATUS "=== Setting up library paths for Ubuntu ===")

    # CUDA 架构仅 x86+NVIDIA 平台需要；RK（aarch64）构建下跳过
    if(DVSTREAM_USE_CUDA)
        set(CMAKE_CUDA_ARCHITECTURES "86;89;120")
    endif()

    # FFmpeg：
    #   - RK 交叉编译默认指向 ffmpeg-rockchip 安装树（由 toolchain 文件的 RK_FFMPEG_ROOT 提供）
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

    # OpenCV（CONFIG 模式查找，供 FindOpenCV.cmake 使用）：
    if(NOT DEFINED OPENCV_ROOT_DIR)
        if(VSTREAM_USE_ROCKCHIP AND RK_OPENCV_ROOT)
            set(OPENCV_ROOT_DIR "${RK_OPENCV_ROOT}" CACHE PATH "Folder contains OpenCV (SDK prebuilt)")
            message(STATUS "OPENCV_ROOT_DIR (rockchip): ${OPENCV_ROOT_DIR}")
        endif()
    else()
        message(STATUS "OPENCV_ROOT_DIR (user defined): ${OPENCV_ROOT_DIR}")
    endif()

    message(STATUS "=== Library paths setup complete ===")
else()
    message(STATUS "SetupLibraryPaths.cmake: Not on Linux, skipping automatic path setup")
endif()
