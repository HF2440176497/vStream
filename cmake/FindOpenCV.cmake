include(FindPackageHandleStandardArgs)

set(OPENCV_ROOT_DIR "" CACHE PATH "Folder contains OpenCV (set if manually compiled)")

# PATH_SUFFIXES 兼容多种布局：
#   cmake/opencv4                        —— 独立安装树（prefix/cmake/opencv4）
#   lib/cmake/opencv4                    —— 标准 prefix 安装（prefix/lib/cmake/opencv4）
#   lib/aarch64-linux-gnu/cmake/opencv4  —— SDK sysroot 的 Debian 布局
#                                            （RK_OPENCV_ROOT 默认指向 sysroot）
#   usr/lib/...                          —— 用户把 ROOT 直接指到 sysroot 根的情况
if(OPENCV_ROOT_DIR AND NOT OPENCV_DIR)
    find_file(OpenCV_CONFIG_FILE OpenCVConfig.cmake
        PATHS ${OPENCV_ROOT_DIR}
        PATH_SUFFIXES
            cmake/opencv4
            cmake
            lib/cmake/opencv4
            lib/aarch64-linux-gnu/cmake/opencv4
            usr/lib/cmake/opencv4
            usr/lib/aarch64-linux-gnu/cmake/opencv4)
    if(OpenCV_CONFIG_FILE)
        get_filename_component(OPENCV_DIR "${OpenCV_CONFIG_FILE}" DIRECTORY)
        message(STATUS "Found OpenCV config in: ${OPENCV_DIR}")
    endif()
endif()


if(OPENCV_DIR)
    # find_package 会在 OPENCV_DIR 中查找 cmake 配置文件
    message(STATUS "Looking for OpenCV using config: ${OPENCV_DIR}")
    find_package(OpenCV REQUIRED
        COMPONENTS core imgproc features2d imgcodecs videoio
        CONFIG)
elseif(OPENCV_ROOT_DIR)
    message(STATUS "Looking for OpenCV in: ${OPENCV_ROOT_DIR}")
    find_package(OpenCV REQUIRED
        COMPONENTS core imgproc features2d imgcodecs videoio
        PATHS ${OPENCV_ROOT_DIR}
        CONFIG)
else()
    message(STATUS "Looking for OpenCV in system paths")
    find_package(OpenCV REQUIRED
        COMPONENTS core imgproc features2d imgcodecs videoio)
endif()


if(OpenCV_FOUND)
    find_package_handle_standard_args(OpenCV
        REQUIRED_VARS OpenCV_INCLUDE_DIRS OpenCV_LIBRARIES
        VERSION_VAR OpenCV_VERSION)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    message(STATUS "  Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCV_LIBRARIES}")
    if(NOT OpenCV_LIBS)
        set(OpenCV_LIBS ${OpenCV_LIBRARIES})
    endif()
    mark_as_advanced(OPENCV_ROOT_DIR OPENCV_DIR)
else()
    message(FATAL_ERROR "OpenCV not found. Please set OPENCV_ROOT_DIR or install OpenCV.")
endif()
