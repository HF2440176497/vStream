#!/usr/bin/env python3
"""从 PP-OCRv6 导出的 inference.yml 提取 PostProcess.character_dict，生成 C++ ReadDict 可用的 label 文件。

用法:
    python3 extract_ppocrv6_dict.py <inference.yml> <ppocr_v6_keys.txt>

产物说明:
    - 每行一个字符, LF 换行(C++ getline 依赖 LF, 不能是 CRLF)
    - 不含 blank 和空格: C++ 侧 Post_PPOCRv6_rec_Obj 会在头部插 blank、尾部追加空格
    - 校验: 行数 + 2 应等于模型输出最后一维 (PP-OCRv6 为 18708 + 2 = 18710)
"""
import sys

import yaml


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    yml_path, out_path = sys.argv[1], sys.argv[2]

    with open(yml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chars = data["PostProcess"]["character_dict"]

    # yaml 可能把 '1'/'on' 这类字符解析成 int/bool, 统一转回字符串并告警
    suspicious = [(i, c) for i, c in enumerate(chars) if not isinstance(c, str)]
    for i, c in suspicious:
        print(f"WARNING: entry {i} parsed as {type(c).__name__}: {c!r}")
    chars = [str(c) for c in chars]

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(chars) + "\n")

    print(f"dict chars written: {len(chars)}")
    print(f"expected model output cols = {len(chars)} + blank(1) + space(1) = {len(chars) + 2}")


if __name__ == "__main__":
    main()
