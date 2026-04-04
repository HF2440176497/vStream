cmake_minimum_required(VERSION 3.13)

if(TARGET vstream_core)
    add_executable(cnstream_inspect cnstream_inspect.cpp)
    target_link_libraries(cnstream_inspect 
        vstream_core
        gflags
        glog
    )
else()
    message(STATUS "vstream_core target not found - skipping build tools")
endif()

