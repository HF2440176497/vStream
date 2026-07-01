# source 此文件即可使用 vstream
# 不设 LD_LIBRARY_PATH —— 第三方依赖由 /etc/ld.so.conf.d/vstream.conf 自适应解析
#   （优先复用目标机已有库，回退到 dev/lib 打包副本）
_prefix=/usr/local/vstream
export PATH="${_prefix}/bin:${PATH}"
export PYTHONPATH="${_prefix}/lib:${PYTHONPATH}"
unset _prefix
