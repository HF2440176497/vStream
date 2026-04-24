cmake_minimum_required(VERSION 3.13)

set(DEMO_SOURCES
  json_parse_demo.cpp
  libyuv_demo.cpp
)

foreach(demo_source ${DEMO_SOURCES})
  get_filename_component(demo_name ${demo_source} NAME_WE)
  add_executable(${demo_name} ${demo_source})
  target_include_directories(${demo_name}
   PRIVATE
    ${VSTREAM_TOOLS_3RDPARTY_INCLUDE}
  )
  target_link_libraries(${demo_name} 
   PUBLIC
    ${VSTREAM_TOOLS_DEPENDENCIES}
   PRIVATE
    ${VSTREAM_TOOLS_3RDPARTY_LIBS}
  )
endforeach()

if(VSTREAM_USE_CUDA)
  add_subdirectory(cuda)
  add_subdirectory(trt)
endif()