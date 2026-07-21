include(FindPackageHandleStandardArgs)

set(OpenCV_ROOT_DIR "" CACHE PATH "Folder contains OpenCV (set if manually compiled)")

if(OpenCV_ROOT_DIR AND NOT OpenCV_DIR)
    find_file(OpenCV_CONFIG_FILE OpenCVConfig.cmake
        PATHS ${OpenCV_ROOT_DIR}
        PATH_SUFFIXES cmake/opencv4 cmake)
    if(OpenCV_CONFIG_FILE)
        get_filename_component(OpenCV_DIR "${OpenCV_CONFIG_FILE}" DIRECTORY)
        message(STATUS "Found OpenCV config in: ${OpenCV_DIR}")
    endif()
endif()


if(OpenCV_DIR)
    # find_package 会在 OpenCV_DIR 中查找 cmake 配置文件
    message(STATUS "Looking for OpenCV using config: ${OpenCV_DIR}")
    find_package(OpenCV REQUIRED 
        COMPONENTS core imgproc features2d imgcodecs videoio
        CONFIG)
elseif(OpenCV_ROOT_DIR)
    message(STATUS "Looking for OpenCV in: ${OpenCV_ROOT_DIR}")
    find_package(OpenCV REQUIRED 
        COMPONENTS core imgproc features2d imgcodecs videoio
        PATHS ${OpenCV_ROOT_DIR})
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
    mark_as_advanced(OpenCV_ROOT_DIR OpenCV_DIR)
else()
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_ROOT_DIR or install OpenCV.")
endif()

