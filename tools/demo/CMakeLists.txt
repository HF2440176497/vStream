cmake_minimum_required(VERSION 3.13)

set(DEMO_SOURCES
    json_parse_demo.cpp
    libyuv_demo.cpp
)

foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    add_executable(${demo_name} ${demo_source})
    add_dependencies(${demo_name} ${TOOLS_DEPENDENCIES})
    target_link_libraries(${demo_name} ${TOOLS_3RDPARTY_LIBS} ${TOOLS_DEPENDENCIES} rt dl pthread)
endforeach()

if(NVIDIA)
  add_subdirectory(cuda)
  add_subdirectory(trt)
endif()