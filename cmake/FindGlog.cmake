#COPYRIGHT
#
#All contributions by the University of California:
#Copyright (c) 2014-2017 The Regents of the University of California (Regents)
#All rights reserved.
#
#All other contributions:
#Copyright (c) 2014-2017, the respective contributors
#All rights reserved.
#
#Caffe uses a shared copyright model: each contributor holds copyright over
#their contributions to Caffe. The project versioning records all such
#contribution and copyright details. If a contributor wants to further mark
#their specific copyright on a particular contribution, they should indicate
#their copyright solely in the commit message of the change when it is
#committed.
#
#LICENSE
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met: 
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer. 
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution. 
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#CONTRIBUTION AGREEMENT
#
#By contributing to the BVLC/caffe repository through pull-request, comment,
#or otherwise, the contributor releases their content to the
#license and copyright terms herein.
#

# - Try to find Glog
#
# The following variables are optionally searched for defaults
#  GLOG_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  GLOG_FOUND
#  GLOG_INCLUDE_DIRS
#  GLOG_LIBRARIES
#  GLOG_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")

if(GLOG_ROOT_DIR)
    find_path(GLOG_INCLUDE_DIR glog/logging.h
        PATHS ${GLOG_ROOT_DIR}
        PATH_SUFFIXES include)
else()
    find_path(GLOG_INCLUDE_DIR glog/logging.h)
endif()

if(WIN32)
    if(GLOG_ROOT_DIR)
        find_library(GLOG_LIBRARY_RELEASE
            NAMES glog glog_static libglog libglog_static
            PATHS ${GLOG_ROOT_DIR}
            PATH_SUFFIXES lib Release)

        find_library(GLOG_LIBRARY_DEBUG
            NAMES glogd glog_debug glog libglog libglog_static
            PATHS ${GLOG_ROOT_DIR}
            PATH_SUFFIXES lib Debug)
    else()
        find_library(GLOG_LIBRARY_RELEASE
            NAMES glog glog_static libglog libglog_static)

        find_library(GLOG_LIBRARY_DEBUG
            NAMES glogd glog_debug glog libglog libglog_static)
    endif()

    if(GLOG_LIBRARY_RELEASE AND GLOG_LIBRARY_DEBUG)
        set(GLOG_LIBRARY optimized ${GLOG_LIBRARY_RELEASE} debug ${GLOG_LIBRARY_DEBUG})
    elseif(GLOG_LIBRARY_DEBUG)
        set(GLOG_LIBRARY ${GLOG_LIBRARY_DEBUG})
    elseif(GLOG_LIBRARY_RELEASE)
        set(GLOG_LIBRARY ${GLOG_LIBRARY_RELEASE})
    endif()
else()
    if(GLOG_ROOT_DIR)
        find_library(GLOG_LIBRARY
            NAMES glog
            PATHS ${GLOG_ROOT_DIR}
            PATH_SUFFIXES lib lib64)
    else()
        find_library(GLOG_LIBRARY glog)
    endif()
endif()

find_package_handle_standard_args(Glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

if(GLOG_FOUND)
  set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
  set(GLOG_LIBRARIES ${GLOG_LIBRARY})
  # glog 0.7.0+ requires GLOG_USE_GLOG_EXPORT to be defined when consuming the library
  add_definitions(-DGLOG_USE_GLOG_EXPORT)
  message(STATUS "Found glog    (include: ${GLOG_INCLUDE_DIR}, library: ${GLOG_LIBRARY})")
  mark_as_advanced(GLOG_ROOT_DIR GLOG_LIBRARY_RELEASE GLOG_LIBRARY_DEBUG
                                 GLOG_LIBRARY GLOG_INCLUDE_DIR)
endif()
