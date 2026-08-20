# cmake/GatherRuntimeDeps.cmake
# 由 install(SCRIPT ...) 调用，此时已进入 staging 安装目录
#
# 设计要点：
#   1. 软链接：所有 .so 用 REALPATH 解析到真实文件，再按基名 glob 整条 SONAME 链拷贝
#   2. 循环依赖：用 _seen_inodes 集合按 (设备号,inode) 去重，避免无限递归/重复拷贝
#   3. Python/TRT：PRE_EXCLUDE_REGEXES 显式排除，由运行环境提供

# CPACK_SET_DESTDIR=ON 时实际文件在 $ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/ 下
# 直接 make install 时 DESTDIR 为空，路径退化为 CMAKE_INSTALL_PREFIX
set(_dest_root "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
set(_staging_lib "${_dest_root}/lib")
set(_staging_dev_lib "${_dest_root}/dev/lib")

if(NOT EXISTS "${_staging_lib}")
  return()
endif()

# ---------- 构建 vstream .so 文件列表 ----------
set(_vstream_libs "")
foreach(_lib_name "libvstream_core.so" "libvstream_va.so")
  if(EXISTS "${_staging_lib}/${_lib_name}")
    list(APPEND _vstream_libs "${_staging_lib}/${_lib_name}")
  endif()
endforeach()

file(GLOB _python_modules "${_staging_lib}/vstream.cpython-*.so")
if(_python_modules)
  list(APPEND _vstream_libs ${_python_modules})
endif()

if(EXISTS "${_staging_lib}/libyuv.so")
  list(APPEND _vstream_libs "${_staging_lib}/libyuv.so")
endif()

if(NOT _vstream_libs)
  message(STATUS "[GatherRuntimeDeps] No vstream libraries found in ${_staging_lib}, skipping.")
  return()
endif()

# ---------- 解析依赖树 ----------
# file(GET_RUNTIME_DEPENDENCIES 内部已实现依赖图遍历，对循环依赖安全
# （它用已访问集合避免无限循环，结果只列依赖项，不会重复）
# CONFLICTING_DEPENDENCIES_PREFIX 用于检测同名库指向不同路径的冲突情况
#
# 原生构建：DIRECTORIES 只含 _staging_lib（vstream 自身库如 libyuv.so 所在），
# 不含 _staging_dev_lib —— 否则已打包的依赖会解析到 staging 路径（含 DESTDIR），
# 导致 dep-manifest.txt 记录的不是编译期系统路径，postinst 自适应复用会失效。
# 第三方依赖由系统默认搜索路径（ld.so.conf）解析，得到的才是编译期真实路径。
#
# RK 交叉编译：第三方依赖位于 sysroot / ffmpeg-rockchip 安装树 / 仓库内 rknpu2，
# 不在构建机默认搜索路径中，必须显式补充解析目录（变量由 BundleThirdPartyDeps 的 install(CODE) 传入）。
set(_dep_search_dirs "${_staging_lib}")
if(VSTREAM_USE_ROCKCHIP)
  foreach(_d
      "${RK_FFMPEG_ROOT}/lib"
      "${RK_SYSROOT}/usr/lib/aarch64-linux-gnu"
      "${RK_SYSROOT}/usr/lib"
      "${RK_SYSROOT}/lib/aarch64-linux-gnu"
      "${RKNN_LIB_DIR}")
    if(IS_DIRECTORY "${_d}")
      list(APPEND _dep_search_dirs "${_d}")
    endif()
  endforeach()
endif()

# PRE：按 NEEDED 名字过滤，命中即不解析路径（最早期排除）
set(_pre_exclude
    "^libc\\.so"
    "^libstdc\\+\\+\\.so"
    "^libgcc_s\\.so"
    "^libpthread\\.so"
    "^libdl\\.so"
    "^libm\\.so"
    "^librt\\.so"
    "^libnvidia-?tl\\.so"
    "^libcuda\\.so"
    "^libcudart\\.so"
    "^libnpp.*\\.so"
    "^libnvrtc\\.so"
    "^libnvml\\.so"
    "^libnvinfer.*\\.so"
    "^libnvonnxparser\\.so"
    "^libpython.*\\.so")
if(NOT VSTREAM_USE_ROCKCHIP)
  # 原生构建：Python 解释器库由运行环境基础镜像提供
  list(APPEND _pre_exclude "^libpython.*\\.so")
else()
  # RK：librknnrt.so 已通过 BundleThirdPartyDeps 显式打包（known_libs），按名排除防重复
  list(APPEND _pre_exclude "^librknnrt\\.so")
endif()

# gflags/glog：已由 install(TARGETS) 装到 lib/，
# 排除防止依赖解析时从 staging lib/ 命中后重复拷进 dev/lib
if(VSTREAM_INTREE_GFLAGS_GLOG)
  list(APPEND _pre_exclude "^libgflags\\.so" "^libglog\\.so")
endif()

# POST：解析到完整路径后再按路径过滤
#   /lib/ /usr/lib/ /usr/lib64/ 视为系统库，不打包
#   交叉编译解析到的 sysroot 路径不带 /usr 前缀，不在排除之列，随包自包含
set(_post_exclude
    "^/lib/"
    "^/lib64/"
    "^/usr/lib/"
    "^/usr/lib32/"
    "^/usr/lib64/"
    "^/usr/lib/x86_64-linux-gnu/"
    "^/usr/lib/aarch64-linux-gnu/"
    "ld-linux")

file(GET_RUNTIME_DEPENDENCIES
  RESOLVED_DEPENDENCIES_VAR     _resolved
  UNRESOLVED_DEPENDENCIES_VAR   _unresolved
  CONFLICTING_DEPENDENCIES_PREFIX _conflict
  DIRECTORIES                   ${_dep_search_dirs}
  EXECUTABLES                   ${_vstream_libs}
  LIBRARIES                     ${_vstream_libs}
  PRE_EXCLUDE_REGEXES           ${_pre_exclude}
  POST_EXCLUDE_REGEXES          ${_post_exclude}
)

# ---------- 处理冲突依赖 ----------
# 同名库被解析到多个不同路径时，输出告警并取第一个（避免随机选择）
if(DEFINED _conflict_FILENAMES)
  list(LENGTH _conflict_FILENAMES _n)
  message(WARNING "[GatherRuntimeDeps] ${_n} conflicting dependencies detected:")
  foreach(_f ${_conflict_FILENAMES})
    message(WARNING "  ${_f}: ${_conflict_${_f}}")
  endforeach()
endif()

# ---------- 拷贝依赖（处理软链接 + 去重）----------
# 用 (st_dev, st_ino) 元组作为唯一性键，对软链接和硬链接都安全
# 同一物理文件被多条依赖链引用时只拷一次
set(_copied_names "")
set(_seen_inodes "")

file(MAKE_DIRECTORY "${_staging_dev_lib}")

foreach(_dep ${_resolved})
  if(NOT EXISTS "${_dep}")
    continue()
  endif()

  # 解析软链接到真实文件，再回推基名，glob 整条 SONAME 链
  # 例: libopencv_core.so -> libopencv_core.so.412 -> libopencv_core.so.4.12.0
  file(REAL_PATH "${_dep}" _real EXPAND_TILDE)
  get_filename_component(_dir "${_real}" DIRECTORY)
  get_filename_component(_base "${_real}" NAME_WE)  # libopencv_core

  file(GLOB _chain "${_dir}/${_base}.so*")
  foreach(_f ${_chain})
    # 取 (设备号,inode) 元组作为去重键
    # 对同一物理文件被多路径引用、循环依赖回环都生效
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
      continue()  # 同一物理文件已拷贝，跳过（循环依赖的回环在此终止）
    endif()
    list(APPEND _seen_inodes "${_inode}")

    # FOLLOW_SYMLINK_CHAIN 保留整条 SONAME 软链结构
    file(COPY "${_f}" DESTINATION "${_staging_dev_lib}" FOLLOW_SYMLINK_CHAIN)
    list(APPEND _copied_names "${_f}")
  endforeach()
endforeach()

if(_copied_names)
  list(REMOVE_DUPLICATES _copied_names)
  message(STATUS "[GatherRuntimeDeps] copied ${_copied_names}")
endif()
if(_unresolved)
  message(WARNING "[GatherRuntimeDeps] unresolved: ${_unresolved}")
endif()

# ---------- 生成 dep-manifest.txt（供 postinst 自适应复用使用）----------
# 遍历所有已解析依赖，提取 SONAME 与所在目录，写入清单
# postinst 据此判断：目标机是否在编译期同路径下有同 SONAME 库
set(_manifest "# vStream dependency manifest - generated at build time\n")
string(APPEND _manifest "# Format: <soname>\t<build-time-resolved-dir>\n")
string(APPEND _manifest "# postinst checks if <dir>/<soname> exists on target; if so, reuses system lib.\n\n")

set(_manifest_seen "")
foreach(_dep ${_resolved})
  if(NOT EXISTS "${_dep}")
    continue()
  endif()
  # 解析软链到真实文件，取所在目录作为"编译期搜索路径"
  file(REAL_PATH "${_dep}" _real)
  get_filename_component(_dir "${_real}" DIRECTORY)

  # 提取 SONAME（DT_SONAME），用于精确匹配
  # 例: /usr/local/lib/libopencv_core.so.4.12.0 的 SONAME 是 libopencv_core.so.412
  execute_process(
    COMMAND readelf -d "${_real}"
    OUTPUT_VARIABLE _elf
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _readelf_res
  )
  set(_soname "")
  if(_readelf_res EQUAL 0)
    string(REGEX MATCH "SONAME[ \t]+\\[([^]]+)\\]" _m "${_elf}")
    if(_m)
      set(_soname "${CMAKE_MATCH_1}")
    endif()
  endif()
  # 无 SONAME 的库（如某些静态链接的 .so）用文件名作 fallback
  if(NOT _soname)
    get_filename_component(_soname "${_real}" NAME)
  endif()

  # 按 (soname, dir) 去重，避免清单膨胀
  set(_key "${_soname}\t${_dir}")
  if(_key IN_LIST _manifest_seen)
    continue()
  endif()
  list(APPEND _manifest_seen "${_key}")
  string(APPEND _manifest "${_soname}\t${_dir}\n")
endforeach()

file(MAKE_DIRECTORY "${_dest_root}/share/vstream")
file(WRITE "${_dest_root}/share/vstream/dep-manifest.txt" "${_manifest}")
list(LENGTH _manifest_seen _manifest_count)
message(STATUS "[GatherRuntimeDeps] wrote dep-manifest.txt "
               "(${_manifest_count} entries) to ${CMAKE_INSTALL_PREFIX}/share/vstream/")
