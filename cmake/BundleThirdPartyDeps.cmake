# cmake/BundleThirdPartyDeps.cmake
# 把容器内已安装的 OpenCV/FFmpeg/gflags/glog 及其传递依赖
# 拷贝到 staging 目录的 dev/lib 与 dev/include
# Python 与 TensorRT 不打包，由运行环境基础镜像提供

function(vstream_bundle_third_party_deps)

  # ---------- 1. 显式拷贝已知 .so ----------
  # 仅收集自包含模式下需要打包的依赖（不含 Python/TRT）
  set(_known_libs "")
  if(OpenCV_FOUND)
    list(APPEND _known_libs ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
  endif()
  if(FFMPEG_FOUND)
    list(APPEND _known_libs ${FFMPEG_LIBRARIES})
  endif()
  if(RKNN_FOUND)
    list(APPEND _known_libs ${RKNN_LIBRARIES})
  endif()
  # libyuv 已是项目目标，通过 install(TARGETS) 处理，此处不重复

  list(REMOVE_DUPLICATES _known_libs)

  # inode 去重集合：避免同一物理文件经多条软链/路径重复 install
  set(_seen_inodes "")

  foreach(_lib ${_known_libs})
    # _lib 形如 /usr/local/lib/libopencv_core.so（Find 模块解析出的完整路径）
    if(NOT EXISTS "${_lib}")
      continue()
    endif()

    # 解析软链接到真实文件，再回推基名，glob 整条 SONAME 链
    #   libfoo.so -> libfoo.so.1 -> libfoo.so.1.0.0
    # 详见第七节对软链接与循环依赖的处理
    get_filename_component(_real "${_lib}" REALPATH)
    get_filename_component(_dir "${_real}" DIRECTORY)
    get_filename_component(_base "${_real}" NAME_WE)  # 去版本号的基名，如 libopencv_core
    file(GLOB _matched "${_dir}/${_base}.so*")

    # 对每个候选文件按 inode 去重，避免重复 install 同一物理文件
    set(_to_install "")
    foreach(_f ${_matched})
      execute_process(
        COMMAND stat -c "%d:%i" "${_f}"
        OUTPUT_VARIABLE _inode
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _stat_res
      )
      if(NOT _stat_res EQUAL 0)
        continue()
      endif()
      if(_inode IN_LIST _seen_inodes)
        continue()
      endif()
      list(APPEND _seen_inodes "${_inode}")
      list(APPEND _to_install "${_f}")
    endforeach()

    if(_to_install)
      install(FILES ${_to_install} DESTINATION dev/lib)
    endif()
  endforeach()

  # ---------- 2. 拷贝第三方头文件 ----------
  if(OpenCV_INCLUDE_DIRS)
    foreach(_inc ${OpenCV_INCLUDE_DIRS})
      if(IS_DIRECTORY "${_inc}")
        install(DIRECTORY ${_inc}/ DESTINATION dev/include/opencv4)
      endif()
    endforeach()
  endif()
  if(FFMPEG_INCLUDE_DIR AND IS_DIRECTORY "${FFMPEG_INCLUDE_DIR}")
    install(DIRECTORY ${FFMPEG_INCLUDE_DIR}/ DESTINATION dev/include/ffmpeg)
  endif()
  # gflags/glog in-tree 编译，不作为头文件打包；头文件由构建期
  # target 的 INTERFACE_INCLUDE_DIRECTORIES 传播，包内二次开发暂不提供
  if(RKNN_FOUND AND RKNN_INCLUDE_DIRS AND IS_DIRECTORY "${RKNN_INCLUDE_DIRS}")
    install(DIRECTORY ${RKNN_INCLUDE_DIRS}/ DESTINATION dev/include/rknpu2)
  endif()

  # ---------- 3. 传递依赖解析 ----------
  # Python 与 TensorRT 在此阶段被 PRE_EXCLUDE_REGEXES 显式排除
  # 注：install(SCRIPT) 在安装阶段执行，读不到 configure 期的普通变量，
  # 故用 install(CODE) 把相关变量烘焙进脚本后再 include 执行
  install(CODE "
    set(VSTREAM_USE_ROCKCHIP \"${VSTREAM_USE_ROCKCHIP}\")
    set(VSTREAM_INTREE_GFLAGS_GLOG \"${VSTREAM_INTREE_GFLAGS_GLOG}\")
    set(RK_SYSROOT \"${RK_SYSROOT}\")
    set(RK_FFMPEG_ROOT \"${RK_FFMPEG_ROOT}\")
    set(RKNN_LIB_DIR \"${CMAKE_SOURCE_DIR}/3rdparty/rknpu2/lib\")
    include(\"${CMAKE_SOURCE_DIR}/cmake/GatherRuntimeDeps.cmake\")
  ")

endfunction()
