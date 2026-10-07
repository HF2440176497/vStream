# -*- coding: utf-8 -*-

"""
precision_eval 统一入口

子命令（与 model_types.json 里各模型类型的 steps 对应）：
    list      列出已登记的模型类型，以及各自需要跑哪些步骤
    dump      通用：图片 -> corpus（预处理由 adapter 提供）
    ref       通用：corpus -> ONNX 参考输出（onnxruntime）
    engine    通用：corpus -> 引擎输出（vStream ModelValidator / TensorRT）
    compare   通用：数值层 + adapter 决策层 -> 报告

"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from precision_eval.core import artifact, metrics                      # noqa: E402
from precision_eval.core.backend import OnnxRuntimeBackend, TrtBackend  # noqa: E402
from precision_eval.adapters import get_adapter, list_adapters          # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------------------------------------------------
# 公共
# ----------------------------------------------------------------------
def parse_params(args):
    """
    合并 --params-file 与 --set，后者优先级更高。
    """
    params = {}
    if getattr(args, "params_file", ""):
        params.update(artifact.load_json(args.params_file))
    for item in (getattr(args, "set", None) or []):
        if "=" not in item:
            raise ValueError("--set 需要 key=value 形式，收到 %r" % item)
        k, v = item.split("=", 1)
        params[k.strip()] = v.strip()
    return params


def resolve_charset(adapter, params, corpus_dir):
    """
    adapter 需要字符表但用户没给路径时，回退到 corpus/charset.txt。

    这样容器侧只需要 corpus
    """
    key = getattr(adapter, "charset_param", None)
    if not key or params.get(key):
        return params
    fallback = os.path.join(corpus_dir, "charset.txt")
    if os.path.exists(fallback):
        params[key] = fallback
    return params


def load_adapter(model_type, params, corpus_dir=None):
    if corpus_dir:
        probe = get_adapter(model_type, {})
        params = resolve_charset(probe, params, corpus_dir)
    return get_adapter(model_type, params)


def list_images(image_dir, limit=0):
    if not os.path.isdir(image_dir):
        raise IOError("图片目录不存在: %s" % image_dir)
    out = [os.path.join(image_dir, n) for n in sorted(os.listdir(image_dir))
           if n.lower().endswith(IMAGE_EXTS)]
    return out[:limit] if limit > 0 else out


def read_image(path):
    import cv2
    img = cv2.imread(path)
    return img


# ----------------------------------------------------------------------
# list
# ----------------------------------------------------------------------
def cmd_list(args):
    items = list_adapters()
    print("已登记的模型类型（共 %d 种）\n" % len(items))
    for it in items:
        print("  %s" % it["name"])
        print("    说明    : %s" % it["description"])
        print("    需要的输入: %s" % ", ".join(it["required_inputs"]))
        print("    需要执行的步骤: %s" % " -> ".join(it["steps"]))
        if it["models"]:
            print("    已登记模型: %s" % ", ".join(it["models"]))
        print("")
    print("按模型类型跑：")
    print("  python -m precision_eval.cli dump    --model-type <类型> ...")
    print("  python -m precision_eval.cli ref     --model-type <类型> ...")
    print("  python -m precision_eval.cli engine  --model-type <类型> ...")
    print("  python -m precision_eval.cli compare --model-type <类型> ...")
    return 0


# ----------------------------------------------------------------------
# dump：图片 -> corpus
# ----------------------------------------------------------------------
def cmd_dump(args):
    params = parse_params(args)
    adapter = get_adapter(args.model_type, params)

    if not adapter.provides_preprocess:
        print("[error] 模型类型 '%s' 未提供预处理实现，无法从图片生成 corpus。" % args.model_type)
        print("        请改为直接提供 corpus（例如使用生产侧 dump 的张量）。")
        return 2

    corpus_dir = os.path.join(args.out_dir, "corpus")
    images = list_images(args.image_dir, args.limit)
    if not images:
        print("[error] 目录下没有图片:", args.image_dir)
        return 3

    # 字符表随语料一起打包，容器侧不再依赖外部路径
    charset_meta = {}
    key = getattr(adapter, "charset_param", None)
    if key and params.get(key):
        src = params[key]
        if not os.path.exists(src):
            print("[error] 字符表不存在:", src)
            return 4
        dst = os.path.join(corpus_dir, "charset.txt")
        os.makedirs(corpus_dir, exist_ok=True)
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
        charset_meta = {
            "charset_file": "charset.txt",
            "charset_sha256": artifact.sha256_file(dst),
            "charset_source": os.path.abspath(src),
        }
        print("[info] 字符表已打包进 corpus: %s" % dst)

    n_ok, n_skip = 0, 0
    for i, path in enumerate(images):
        cid = os.path.splitext(os.path.basename(path))[0]
        img = read_image(path)
        if img is None:
            print("[warn] 读图失败，跳过: %s" % path)
            n_skip += 1
            continue
        try:
            arr = adapter.preprocess(img)
        except Exception as e:
            print("[warn] 预处理失败，跳过 %s: %s" % (cid, e))
            n_skip += 1
            continue

        meta = artifact.new_provenance(
            src_image=os.path.abspath(path),
            src_size_wh=[int(img.shape[1]), int(img.shape[0])],
            model_type=args.model_type,
        )
        meta.update(charset_meta)
        artifact.write_artifact(corpus_dir, cid, arr, meta)
        n_ok += 1
        print("[%d/%d] %s -> %r" % (i + 1, len(images), cid, list(arr.shape)))

    artifact.dump_json(os.path.join(corpus_dir, "corpus_manifest.json"), {
        "schema_version": artifact.SCHEMA_VERSION,
        "model_type": args.model_type,
        "image_dir": os.path.abspath(args.image_dir),
        "num_cases": n_ok,
        "num_skipped": n_skip,
        "cases": artifact.list_cases(corpus_dir),
    })
    print("\n[done] corpus: %s（%d 个样本，跳过 %d）" % (corpus_dir, n_ok, n_skip))
    return 0


# ----------------------------------------------------------------------
# ref / engine：corpus -> 某后端输出
# ----------------------------------------------------------------------
def _run_backend(adapter, backend, corpus_dir, out_dir, stage, extra_meta):
    cases = artifact.list_cases(corpus_dir)
    if not cases:
        print("[error] corpus 目录下没有样本:", corpus_dir)
        return 2, []

    warns = adapter.validate({
        "input_shapes": [backend.input_shape()],
        "output_shapes": [backend.output_shape()],
        "backend": backend.name,
    })
    for w in warns:
        print("[warn] 自检: %s" % w)

    results = []
    for i, cid in enumerate(cases):
        arr, cmeta = artifact.read_artifact(corpus_dir, cid)
        try:
            out = backend.run(arr)
        except Exception as e:
            print("[warn] %s 推理失败，跳过: %s" % (cid, e))
            continue

        dec = None
        if adapter.provides_decision:
            dec = adapter.decision(out)

        meta = artifact.new_provenance(
            stage=stage,
            model_type=adapter.name,
            src_image=cmeta.get("src_image", ""),
            input_shape=list(arr.shape),
        )
        meta.update(extra_meta)
        if dec is not None:
            meta["decision"] = dec
            meta["decision_digest"] = adapter.decision_digest(dec)
        artifact.write_artifact(out_dir, cid, out, meta)

        results.append({"case_id": cid, "decision": dec})
        if dec is not None:
            print("[%d/%d] %s | %s" % (i + 1, len(cases), cid, adapter.decision_digest(dec)))
        else:
            print("[%d/%d] %s | 已存输出 %r" % (i + 1, len(cases), cid, list(out.shape)))

    return 0, results


def cmd_ref(args):
    params = parse_params(args)
    adapter = load_adapter(args.model_type, params, args.corpus_dir)

    providers = [p for p in (args.providers or "").split(",") if p]
    backend = OnnxRuntimeBackend(args.onnx, providers or None)
    desc = backend.describe()
    print("[info] onnxruntime 后端: providers=%s" % desc["providers"])
    print("[info] input=%s output=%s" % (desc["input_shapes"], desc["output_shapes"]))

    ref_dir = os.path.join(args.out_dir, "ref")
    extra = {
        "onnx_path": os.path.abspath(args.onnx),
        "onnx_sha256": artifact.sha256_file(args.onnx),
        "backend": desc["backend"],
        "providers": desc["providers"],
        "output_shape": desc["output_shapes"][0] if desc["output_shapes"] else [],
    }
    code, results = _run_backend(adapter, backend, args.corpus_dir, ref_dir, "ref", extra)
    backend.close()

    artifact.dump_json(os.path.join(ref_dir, "ref_manifest.json"), {
        "stage": "ref", "model_type": adapter.name, "num_cases": len(results), "provenance": extra,
    })
    print("\n[done] ref: %s（%d 个样本）" % (ref_dir, len(results)))
    return code


def cmd_engine(args):
    params = parse_params(args)
    adapter = load_adapter(args.model_type, params, args.corpus_dir)

    backend = TrtBackend(args.engine, args.device, args.device_id, args.input_ordered_index)
    desc = backend.describe()
    print("[info] engine 后端: %s" % desc)

    tag = ("_" + args.tag) if args.tag else ""
    engine_dir = os.path.join(args.out_dir, "engine" + tag)
    extra = {
        "engine_path": os.path.abspath(args.engine),
        "engine_sha256": artifact.sha256_file(args.engine),
        "backend": desc["backend"],
        "device": desc["device"],
        "device_id": desc["device_id"],
        "tag": args.tag,
        "output_shape": desc["output_shapes"][0] if desc["output_shapes"] else [],
    }
    code, results = _run_backend(adapter, backend, args.corpus_dir, engine_dir, "engine", extra)

    artifact.dump_json(os.path.join(engine_dir, "engine_manifest.json"), {
        "stage": "engine", "model_type": adapter.name, "tag": args.tag,
        "num_cases": len(results), "provenance": extra,
    })
    print("\n[done] engine: %s（%d 个样本）" % (engine_dir, len(results)))
    return code


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
def cmd_compare(args):
    params = parse_params(args)
    adapter = load_adapter(args.model_type, params, args.corpus_dir)

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.corpus_dir)), "compare")
    os.makedirs(out_dir, exist_ok=True)

    ref_cases = set(artifact.list_cases(args.ref_dir))
    eng_cases = set(artifact.list_cases(args.engine_dir))
    common = sorted(ref_cases & eng_cases)
    if not common:
        print("[error] ref 与 engine 目录没有交集，检查 case_id 是否对齐")
        return 2

    tol = args.tol_max_abs
    rows = []
    n_numeric_fail = 0

    for cid in common:
        ref_arr, ref_meta = artifact.read_artifact(args.ref_dir, cid)
        eng_arr, eng_meta = artifact.read_artifact(args.engine_dir, cid)

        row = {"case_id": cid, "src_image": ref_meta.get("src_image", "")}
        row["numeric"] = metrics.numeric_compare(ref_arr, eng_arr)
        row["numeric_pass"] = bool(row["numeric"].get("shape_match")
                                   and row["numeric"]["max_abs"] <= tol)
        if not row["numeric_pass"]:
            n_numeric_fail += 1

        if adapter.provides_decision:
            row["decision"] = adapter.decision_compare(ref_meta.get("decision"),
                                                       eng_meta.get("decision"))
        rows.append(row)

    # 决策层汇总完全交给 adapter（自描述），cli 只消费标准契约键
    dec_summary = adapter.decision_summary(rows) if adapter.provides_decision else {}
    n_decision_mismatch = dec_summary.get("num_decision_mismatch", 0)
    n_structural = dec_summary.get("num_structural_flips", 0)

    summary = {
        "model_type": adapter.name,
        "num_cases": len(rows),
        "tol_max_abs": tol,
        "num_numeric_fail": n_numeric_fail,
        "worst_max_abs": max(r["numeric"].get("max_abs", 0.0) for r in rows),
        "worst_case": max(rows, key=lambda r: r["numeric"].get("max_abs", 0.0))["case_id"],
        "ref_onnx_sha256": ref_meta.get("onnx_sha256", ""),
        "engine_sha256": eng_meta.get("engine_sha256", ""),
    }
    summary.update(dec_summary)   # 标准契约键 + adapter 的任意扩展键

    if n_numeric_fail == 0 and n_decision_mismatch == 0:
        verdict = "PASS：数值层全部在阈值内，结论完全一致。转换未引入可观测差异。"
    elif n_numeric_fail == 0 and n_decision_mismatch > 0:
        verdict = ("数值一致但结论出现翻转：差异量级很小，翻转发生在决策接近的位置，"
                   "属「决策边界脆弱」")
    elif n_numeric_fail > 0 and n_structural > 0:
        verdict = ("数值差异超阈值且存在大 margin 翻转：后端存在结构性差异。"
                   "优先排查构建过程")
    elif n_numeric_fail > 0:
        verdict = "数值差异超阈值：需要继续定位差异来源。"
    else:
        verdict = "无结论。"

    artifact.dump_json(os.path.join(out_dir, "report.json"),
                       {"summary": summary, "verdict": verdict, "cases": rows})
    write_markdown(os.path.join(out_dir, "report.md"), summary, verdict, rows, adapter)
    print(verdict)
    print("\n报告: %s" % os.path.join(out_dir, "report.md"))
    print("最差 max|Δ| = %.3e (%s)" % (summary["worst_max_abs"], summary["worst_case"]))
    return 0


def write_markdown(path, summary, verdict, rows, adapter):
    L = []
    L.append("# 模型精度对比报告\n")
    L.append("- 模型类型: `%s`" % summary["model_type"])
    L.append("- 样本数: %d" % summary["num_cases"])
    L.append("- ONNX sha256: `%s`" % (summary["ref_onnx_sha256"] or "-"))
    L.append("- engine sha256: `%s`" % (summary["engine_sha256"] or "-"))
    L.append("")
    L.append("## 结论\n")
    L.append(verdict + "\n")
    L.append("## 汇总\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append("| 数值超阈值样本 | %d |" % summary["num_numeric_fail"])
    if "num_decision_mismatch" in summary:
        L.append("| 结论不一致样本 | %d |" % summary["num_decision_mismatch"])
        L.append("| 脆弱翻转步 | %d |" % summary["num_fragile_flips"])
        L.append("| 结构性翻转步 | %d |" % summary["num_structural_flips"])
    L.append("| 最差 max\\|Δ\\| | %.3e（%s） |" % (summary["worst_max_abs"], summary["worst_case"]))
    L.append("")

    L.append("## 数值层（通用）\n")
    L.append("| case | max\\|Δ\\| | mean\\|Δ\\| | rms | rel_l2 | 余弦 | 通过 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["numeric"].get("max_abs", 0.0)):
        n = r["numeric"]
        if not n.get("shape_match", True):
            L.append("| %s | 形状不一致 %r vs %r | | | | | ✗ |"
                     % (r["case_id"], n.get("ref_shape"), n.get("eng_shape")))
            continue
        L.append("| %s | %.3e | %.3e | %.3e | %.3e | %.6f | %s |" % (
            r["case_id"], n["max_abs"], n["mean_abs"], n["rms"], n["rel_l2"], n["cosine"],
            "✓" if r["numeric_pass"] else "✗"))
    L.append("")

    # 决策层小节由 adapter 自描述渲染，cli 不感知任何模型字段
    L.extend(adapter.render_decision_report(rows))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------
# argparse
# ----------------------------------------------------------------------
def add_common(p):
    p.add_argument("--model-type", required=True, help="模型类型，见 `list` 子命令")
    p.add_argument("--set", action="append", default=[],
                   help="adapter 参数，key=value，可重复；优先级高于 --params-file")
    p.add_argument("--params-file", default="", help="adapter 参数 JSON 文件")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="precision_eval",
        description="ONNX / TensorRT 模型部署精度评估（通用层 + 模型适配器）")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="列出已登记的模型类型与需要执行的步骤")

    p = sub.add_parser("dump", help="图片 -> corpus（预处理由 adapter 提供）")
    add_common(p)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="只取前 N 张，0 表示全部")

    p = sub.add_parser("ref", help="corpus -> ONNX 参考输出（onnxruntime）")
    add_common(p)
    p.add_argument("--onnx", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--providers", default="", help="如 CPUExecutionProvider 或 CUDAExecutionProvider")

    p = sub.add_parser("engine", help="corpus -> 引擎输出（vStream ModelValidator）")
    add_common(p)
    p.add_argument("--engine", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--input-ordered-index", type=int, default=0)
    p.add_argument("--tag", default="", help="引擎变体标签，如 tf32off")

    p = sub.add_parser("compare", help="数值层 + 决策层对比，出报告")
    add_common(p)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--engine-dir", required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--tol-max-abs", type=float, default=1e-3,
                   help="数值层阈值，FP32 建议 1e-3，严格 1e-4")

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 0

    handlers = {
        "list": cmd_list,
        "dump": cmd_dump,
        "ref": cmd_ref,
        "engine": cmd_engine,
        "compare": cmd_compare,
    }
    try:
        return handlers[args.cmd](args)
    except (KeyError, IOError, ValueError, RuntimeError, ImportError) as e:
        print("[error] %s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
