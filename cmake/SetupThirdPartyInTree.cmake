# cmake/SetupThirdPartyInTree.cmake
# gflags / glog 在所有平台统一从 3rdparty 源码
# 不再依赖系统安装，保证各平台版本一致。
#
# 手动编译参考：
#   gflags: cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON \
#                 -DINSTALL_HEADERS=ON -DINSTALL_SHARED_LIBS=ON -DINSTALL_STATIC_LIBS=ON ..
#   glog:   cmake -DBUILD_SHARED_LIBS=ON ..
#
# 实现要点：
#   1. add_subdirectory 引入 gflags / glog（共享库，gflags target 为 gflags_shared）
#   2. 预置 FindGFlags/FindGlog 的结果变量（GFLAGS_*/GLOG_* 缓存变量），
#      framework / modules / python / tools 中的 include(FindGFlags.cmake)
#      会因缓存变量已存在而跳过系统搜索，直接复用 in-tree target 链接
#   3. 打包时（VSTREAM_PACKAGE=ON）随 vstream 安装到 lib/

# ---------- gflags（共享库） ----------
set(BUILD_SHARED_LIBS    ON  CACHE BOOL "gflags: build shared" FORCE)
set(BUILD_STATIC_LIBS    OFF CACHE BOOL "gflags: build static" FORCE)
# 子项目自身不执行安装（由 VSTREAM_PACKAGE 统一安装到 lib/）
set(INSTALL_HEADERS      OFF CACHE BOOL "gflags: install headers" FORCE)
set(INSTALL_SHARED_LIBS  OFF CACHE BOOL "gflags: install shared libs" FORCE)
set(INSTALL_STATIC_LIBS  OFF CACHE BOOL "gflags: install static libs" FORCE)
add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/gflags EXCLUDE_FROM_ALL)

# ---------- glog（共享库） ----------
# 不使用外部 gflags（glog 侧未链接 gflags 时为独立库，主工程各自链接 gflags）
set(WITH_GFLAGS OFF CACHE BOOL "glog: link gflags" FORCE)
add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/glog EXCLUDE_FROM_ALL)

# gflags/glog 之后恢复默认值
# （googletest / backward-cpp 均按静态库构建）
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared libraries" FORCE)

# ---------- 预置 Find 模块结果变量 ----------
# GFLAGS_LIBRARY / GLOG_LIBRARY 填 target 名，target_link_libraries 直接链接
# in-tree target，头文件搜索路径由 target 的 INTERFACE_INCLUDE_DIRECTORIES 传播
set(GFLAGS_INCLUDE_DIR "${VSTREAM_ROOT_PATH}/3rdparty/gflags/src" CACHE FILEPATH "gflags include dir (in-tree)" FORCE)
set(GFLAGS_LIBRARY     "gflags_shared"                             CACHE FILEPATH "gflags library (in-tree target)" FORCE)
set(GFLAGS_FOUND TRUE)
message(STATUS "gflags building from 3rdparty (target: gflags_shared)")

set(GLOG_INCLUDE_DIR "${VSTREAM_ROOT_PATH}/3rdparty/glog/src" CACHE FILEPATH "glog include dir (in-tree)" FORCE)
set(GLOG_LIBRARY     "glog"                                   CACHE FILEPATH "glog library (in-tree target)" FORCE)
set(GLOG_FOUND TRUE)
message(STATUS "glog building from 3rdparty (target: glog)")

# 标记：供打包层识别（Bundle 不再从系统路径收 gflags/glog 头文件，
# Gather 不把已装到 lib/ 的 gflags/glog 重复收进 dev/lib）
set(VSTREAM_INTREE_GFLAGS_GLOG TRUE)

if(VSTREAM_PACKAGE)
  if(TARGET gflags_shared)
    install(TARGETS gflags_shared
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
    )
  endif()
  if(TARGET glog)
    install(TARGETS glog
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
    )
  endif()
endif()
