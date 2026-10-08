# -*- coding: utf-8 -*-
"""PP-OCR 系列文本识别模型适配器

参数（通过 --set / --params-file 传入）：
  charset_path     字符表文件路径（部署包 postproc 配置里的 label_path）。决策层必需。
  input_shape      模型输入形状，默认 "1,3,48,320"。预处理用其 H/W。
  fragile_margin   判定"决策边界脆弱"的 margin 阈值，默认 1.0。


索引约定（两处一致）：
  0        blank 占位
  1..N     字符表第 1..N 行
  N+1      空格
"""

import math

import numpy as np

from ..core.metrics import topk_stats
from .base import ModelAdapter


class PPOCRRecAdapter(ModelAdapter):
    name = "ppocr_rec"
    description = "PP-OCR 系列文本识别模型适配器"
    required_inputs = ["onnx", "engine", "charset", "images"]

    provides_preprocess = True
    provides_decision = True
    charset_param = "charset_path"

    def __init__(self, params=None):
        super(PPOCRRecAdapter, self).__init__(params)

        shape = str(self.params.get("input_shape", "1,3,48,320"))
        dims = [int(x) for x in shape.replace(" ", "").split(",") if x != ""]
        if len(dims) == 4:
            self.input_height, self.input_width = dims[2], dims[3]
        elif len(dims) == 3:
            self.input_height, self.input_width = dims[1], dims[2]
        else:
            raise ValueError("input_shape 需为 4 维或 3 维，如 1,3,48,320，收到 %r" % shape)

        self.fragile_margin = float(self.params.get("fragile_margin", 1.0))

        self.chars = None
        # 字符表构造方式因模型而异：PP-OCRv6 需补一个空格位（18708+1+1=18710）
        # PP-OCRv3 不补（6624+1=6625）
        # 设置为 auto 则根据模型类别数自动判断
        self.append_space = self.params.get("append_space", "auto")
        self._raw_chars = None
        charset_path = self.params.get("charset_path", "")
        if charset_path:
            self.load_charset(charset_path)

    # ------------------------------------------------------------------
    # 字符表
    # ------------------------------------------------------------------
    def load_charset(self, path):
        """
        读取字符表原始行，并按 append_space 规则构造
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n").rstrip("\r") for line in f]
        self._raw_chars = lines
        self.chars = self._build_charset(self._default_append_space())
        return self.chars

    def _build_charset(self, append_space):
        chars = list(self._raw_chars)
        if append_space:
            chars.append(" ")
        chars.insert(0, "#")   # index 0 = blank 占位，恒不输出
        return chars

    def _flag(self):
        """
        append_space 参数归一化：True / False / None(auto)
        """
        v = self.append_space
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        return None

    def _default_append_space(self):
        f = self._flag()
        return True if f is None else f

    def resolve_for_classes(self, num_classes):
        """
        用模型类别数选定字符表变体，返回 (chars, matched)

        auto 模式下依次尝试「补空格 / 不补空格」，取类别数吻合的那个。
        """
        if self._raw_chars is None:
            return None, False

        f = self._flag()

        # 显式指定了 append_space 参数
        if f is not None:
            chars = self._build_charset(f)
            return chars, len(chars) == num_classes

        for flag in (True, False):
            chars = self._build_charset(flag)
            if len(chars) == num_classes:
                self.chars = chars
                return chars, True

        self.chars = self._build_charset(True)
        return self.chars, False

    def _require_charset(self):
        if self.chars is None:
            raise ValueError(
                "ppocr_rec 需要 charset_path（部署包 postproc 配置里的 label_path）。"
                "请通过 --set charset_path=... 或 --params-file 指定。"
            )

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------
    def preprocess(self, image_bgr, meta_in=None):
        """
        输入：已裁剪的文字行图（BGR，uint8）
        输出：[1, 3, H, W] float32，保持 BGR 通道序
        """
        import cv2

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图为空")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("期望 BGR 三通道图，收到 shape=%r" % (image_bgr.shape,))

        img_h, img_w = self.input_height, self.input_width
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("输入图尺寸非法: %dx%d" % (w, h))

        ratio = w / float(h)
        resize_w = int(math.ceil(img_h * ratio))
        if resize_w > img_w:
            resize_w = img_w
        if resize_w <= 0:
            resize_w = 1

        resized = cv2.resize(image_bgr, (resize_w, img_h), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32)
        resized = resized / 255.0
        resized = resized - 0.5
        resized = resized / 0.5

        padded = np.zeros((img_h, img_w, 3), dtype=np.float32)
        padded[:, 0:resize_w, :] = resized

        chw = padded.transpose(2, 0, 1)          # HWC -> CHW，保持 BGR
        return chw[np.newaxis, ...].copy()       # [1,3,H,W]

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decision(self, logits):
        self._require_charset()
        a = np.asarray(logits, dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        top1, top1_val, _top2, margin = topk_stats(a)

        kept = []
        last = None
        for i in range(len(top1)):
            idx = int(top1[i])
            if idx > 0 and not (i > 0 and idx == last):
                kept.append(i)
            last = idx

        chars = self.chars
        text = "".join(chars[int(top1[i])] for i in kept if int(top1[i]) < len(chars))
        score = float(np.mean(top1_val[kept])) if kept else 0.0

        return {
            "text": text,
            "score": score,
            "top1": [int(x) for x in top1],
            "margin": [float(x) for x in margin],
            "kept_steps": kept,
            "chars_at": [chars[int(x)] if int(x) < len(chars) else "<oob>" for x in top1],
        }

    def decision_digest(self, dec):
        if not dec:
            return ""
        return "'%s' (score=%.4f, %d 字符)" % (dec["text"], dec["score"], len(dec["text"]))

    # ------------------------------------------------------------------
    # 决策层对比
    # ------------------------------------------------------------------
    def decision_compare(self, ref_dec, eng_dec):
        if not ref_dec or not eng_dec:
            return {}

        ref_top1 = np.asarray(ref_dec["top1"], dtype=np.int64)
        eng_top1 = np.asarray(eng_dec["top1"], dtype=np.int64)
        n = min(len(ref_top1), len(eng_top1))
        ref_top1, eng_top1 = ref_top1[:n], eng_top1[:n]

        # 定位到
        flip_steps = np.where(ref_top1 != eng_top1)[0].tolist()
        detail = []
        n_fragile = 0
        n_structural = 0
        for t in flip_steps:
            rm = float(ref_dec["margin"][t])
            fragile = rm < self.fragile_margin
            if fragile:
                n_fragile += 1
            else:
                n_structural += 1
            detail.append({
                "step": int(t),
                "ref_char": ref_dec["chars_at"][t] if t < len(ref_dec["chars_at"]) else "?",
                "eng_char": eng_dec["chars_at"][t] if t < len(eng_dec["chars_at"]) else "?",
                "ref_top1": int(ref_top1[t]),
                "eng_top1": int(eng_top1[t]),
                "ref_margin": rm,
                "eng_margin": float(eng_dec["margin"][t]),
                "fragile": bool(fragile),
            })

        ref_text = ref_dec["text"]
        eng_text = eng_dec["text"]
        return {
            "ref_text": ref_text,
            "eng_text": eng_text,
            "text_same": ref_text == eng_text,
            "top1_match_rate": float((ref_top1 == eng_top1).mean()) if n else 0.0,
            "num_flip_steps": len(flip_steps),
            "num_fragile_flips": n_fragile,
            "num_structural_flips": n_structural,
            "fragile_margin": self.fragile_margin,
            "flip_detail": detail,
        }

    # ------------------------------------------------------------------
    # 决策层自描述：汇总 + 渲染（cli 不再感知本模型的任何字段）
    # ------------------------------------------------------------------
    def decision_summary(self, rows):
        dec_rows = [r for r in rows if r.get("decision")]
        n_mismatch = 0
        n_fragile = 0
        n_structural = 0
        for r in dec_rows:
            d = r["decision"]
            if not d.get("text_same", True):
                n_mismatch += 1
            n_fragile += d.get("num_fragile_flips", 0)
            n_structural += d.get("num_structural_flips", 0)
        return {
            "num_decision_mismatch": n_mismatch,
            "num_fragile_flips": n_fragile,
            "num_structural_flips": n_structural,
        }

    def render_decision_report(self, rows):
        dec_rows = [r for r in rows if r.get("decision")]
        if not dec_rows:
            return []

        L = []
        L.append("## 决策层（%s）\n" % self.name)
        L.append("| case | ONNX | engine | 一致 | top1一致率 | 翻转步 |")
        L.append("|---|---|---|---|---|---|")
        for r in dec_rows:
            d = r["decision"]
            L.append("| %s | %s | %s | %s | %.3f | %d |" % (
                r["case_id"], d.get("ref_text", ""), d.get("eng_text", ""),
                "✓" if d.get("text_same") else "✗",
                d.get("top1_match_rate", 0.0), d.get("num_flip_steps", 0)))
        L.append("")

        flips = [(r["case_id"], f) for r in dec_rows for f in r["decision"].get("flip_detail", [])]
        if flips:
            L.append("## 翻转步明细（关键）\n")
            L.append("| case | 步 | ONNX 字符 | engine 字符 | ONNX margin | engine margin | 判定 |")
            L.append("|---|---|---|---|---|---|---|")
            for cid, f in flips:
                L.append("| %s | %d | %s | %s | %.4f | %.4f | %s |" % (
                    cid, f["step"], f["ref_char"], f["eng_char"],
                    f["ref_margin"], f["eng_margin"],
                    "边界脆弱" if f["fragile"] else "结构性"))
            L.append("")
        return L

    # ------------------------------------------------------------------
    # 自检
    # ------------------------------------------------------------------
    def validate(self, model_info):
        warns = []
        out_shapes = model_info.get("output_shapes") or []
        if not out_shapes:
            return warns

        last = out_shapes[0]
        if len(last) != 3:
            warns.append("期望 3 维输出 [1, T, C]，实际 %r" % (last,))
            return warns

        num_classes = int(last[2])
        if self._raw_chars is not None:
            _chars, matched = self.resolve_for_classes(num_classes)
            if not matched:
                warns.append(
                    "字典与模型类别数不匹配：模型 %d 类，字典 %d 行 "
                    "（补空格=%d 类 / 不补空格=%d 类）。请确认 charset_path 与 append_space。"
                    % (num_classes, len(self._raw_chars),
                       len(self._raw_chars) + 2, len(self._raw_chars) + 1))
        elif self.chars is None:
            warns.append("未提供 charset_path，决策层将不可用（只做数值对比）")

        time_steps = int(last[1])
        if self.input_width and time_steps != self.input_width // 8:
            warns.append(
                "时间步 %d 与输入宽 %d 不成 1/8 关系，确认 input_shape 是否正确"
                % (time_steps, self.input_width))
        return warns
