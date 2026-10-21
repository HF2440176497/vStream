# cmake/Packaging.cmake
# 由根 CMakeLists.txt 在 VSTREAM_PACKAGE=ON 时 include
# 负责 CPack DEB 配置、维护脚本注册、第三方依赖收集（自包含模式）

set(CPACK_PACKAGE_NAME "vstream" CACHE STRING "Debian package name")
# 版本来自根目录 VERSION 文件（VSTREAM_VERSION，见顶层 CMakeLists.txt）
set(CPACK_PACKAGE_VERSION "${VSTREAM_VERSION}")
set(CPACK_PACKAGE_RELEASE "1" CACHE STRING "vStream package release number")

# CPackDeb 只认 CPACK_DEBIAN_PACKAGE_RELEASE：
# 不设置时 deb 的 Version 字段会丢掉 -1 后缀（与文件名不一致）
set(CPACK_DEBIAN_PACKAGE_RELEASE "${CPACK_PACKAGE_RELEASE}")
set(CPACK_PACKAGE_CONTACT "Sasha <vstream@example.com>")
set(CPACK_PACKAGE_VENDOR "vStream")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "vStream - video structured analysis framework")
# README.md 可选：缺失时回退到单行描述，否则 cpack 会因打不开文件而失败
if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
  set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
  set(CPACK_RESOURCE_FILE_README  "${CMAKE_SOURCE_DIR}/README.md")
else()
  message(STATUS "README.md not found, using summary as DEB long description")
  set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}")
endif()

# 自动检测架构
# 交叉编译时 dpkg --print-architecture 取到的是构建机架构（amd64），必须按目标平台显式指定
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
  message(STATUS "Cross-compiling for aarch64, DEB architecture forced to arm64")
else()
  execute_process(
    COMMAND dpkg --print-architecture
    OUTPUT_VARIABLE _deb_arch
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _arch_res
  )
  if(_arch_res EQUAL 0 AND _deb_arch)
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${_deb_arch})
  else()
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  endif()
endif()

# DEB 生成器
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://example.com/vstream")
set(CPACK_DEBIAN_PACKAGE_SECTION "video")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")

# deb 文件名：默认 vstream_<VERSION>-<RELEASE>_<ARCH>.deb
# 如需完全自定义（例如追加日期、平台后缀），配置时传:
#   -DVSTREAM_PACKAGE_FILE_NAME=vstream_1.0.0-1_amd64_cuda
if(VSTREAM_PACKAGE_FILE_NAME)
  set(CPACK_PACKAGE_FILE_NAME "${VSTREAM_PACKAGE_FILE_NAME}")
else()
  set(CPACK_PACKAGE_FILE_NAME
      "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
endif()

# 依赖：仅声明系统级依赖，自定义编译的依赖已随包
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "libc6 (>= 2.35), libstdc++6 (>= 12), libgcc-s1 (>= 12)")

# 确保维护脚本有执行权限（configure 阶段执行一次）
set(_deb_scripts
  "${CMAKE_SOURCE_DIR}/cmake/deb/postinst"
  "${CMAKE_SOURCE_DIR}/cmake/deb/prerm"
  "${CMAKE_SOURCE_DIR}/cmake/deb/postrm"
)
foreach(_script ${_deb_scripts})
  if(EXISTS "${_script}")
    execute_process(
      COMMAND chmod +x "${_script}"
      RESULT_VARIABLE _chmod_res
    )
  endif()
endforeach()

# deb 展开后应运行 ldconfig
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${CMAKE_SOURCE_DIR}/cmake/deb/postinst;${CMAKE_SOURCE_DIR}/cmake/deb/prerm;${CMAKE_SOURCE_DIR}/cmake/deb/postrm")

# 让 deb 内文件按 CMAKE_INSTALL_PREFIX 落地（关键：使 /usr/local/vstream 生效）
set(CPACK_SET_DESTDIR ON)
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# 不拆组件，一个 deb 全包
set(CPACK_DEB_COMPONENT_INSTALL OFF)

# ============================================================
# 用户环境 source 脚本（装到 share/vstream/）
# 不设 LD_LIBRARY_PATH —— 第三方依赖由 ld.so.conf.d 自适应解析
# ============================================================
install(FILES ${CMAKE_SOURCE_DIR}/cmake/vstream.sh
  DESTINATION share/vstream
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)

# 版本标记文件（install 时写入，供运维确认当前安装版本）
# 注意：install(CODE) 不自动追加 DESTDIR，需手动拼接 $ENV{DESTDIR}
install(CODE "
  set(_dest_root \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
  file(MAKE_DIRECTORY \"\${_dest_root}/share/vstream\")
  file(WRITE \"\${_dest_root}/share/vstream/VERSION\" \"${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}\n\")
")

# ============================================================
# 第三方依赖收集（自包含模式）
# ============================================================
if(VSTREAM_PACKAGE_BUNDLE_DEPS)
  include(${CMAKE_SOURCE_DIR}/cmake/BundleThirdPartyDeps.cmake)
  vstream_bundle_third_party_deps()
endif()

# 必须在所有 install 规则之后、最末尾调用
include(CPack)
