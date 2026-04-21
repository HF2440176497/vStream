cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -g")

set(CMAKE_MODULE_PATH ${VSTREAM_ROOT_PATH}/cmake)

# ---[ pybind11
set(PYBIND11_PYTHON_VERSION 3)
add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/pybind11)

# ---[ opencv



