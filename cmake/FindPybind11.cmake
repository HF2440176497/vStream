# - Try to find Pybind11
#
# The following variables are optionally searched for defaults
#  PYBIND11_ROOT_DIR:            Base directory where all Pybind11 components are found
#
# The following are set after configuration is done:
#  PYBIND11_FOUND
#  PYBIND11_INCLUDE_DIRS

include(FindPackageHandleStandardArgs)

set(PYBIND11_ROOT_DIR "" CACHE PATH "Folder contains Pybind11")

if(PYBIND11_ROOT_DIR)
    find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h
        PATHS ${PYBIND11_ROOT_DIR}
        PATH_SUFFIXES include)
else()
    find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h)
endif()

# auto set PYBIND11_FOUND
find_package_handle_standard_args(Pybind11 DEFAULT_MSG PYBIND11_INCLUDE_DIR)

if(PYBIND11_FOUND)
    set(PYBIND11_INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR})
    message(STATUS "Found pybind11 (include: ${PYBIND11_INCLUDE_DIR})")
    mark_as_advanced(PYBIND11_INCLUDE_DIR PYBIND11_ROOT_DIR)
endif()
