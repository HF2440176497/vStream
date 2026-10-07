# -*- coding: utf-8 -*-
"""
模型适配器接口

  core/      只回答「两个张量差多少」，不认识任何模型；
  adapters/  回答「这个差异对这个模型意味着什么」，是唯一允许出现
             预处理公式、字符表、CTC、NMS、类别名等概念的地方。

新增模型类型时，只需要：
  1. 在本目录加一个 <model>.py，继承 ModelAdapter；
  2. 在 ../model_types.json 登记（module / class / steps / description）。

"""


class ModelAdapter(object):
    # ---- 元信息（供 cli list 展示）----
    name = "base"
    description = ""
    # 该模型需要的输入物，供用户判断要准备什么
    required_inputs = []

    # ---- 能力声明（cli 据此决定哪些子命令可用）----
    provides_preprocess = False   # 能否把图片转成模型输入张量（决定 dump 是否可用）
    provides_decision = False     # 能否做模型特定的解码与决策层对比

    # 若该模型需要字符表/标签表，填它在 params 里的键名；
    # cli 会据此把它打包进 corpus，并在容器侧自动回退读取。
    charset_param = None

    def __init__(self, params=None):
        self.params = dict(params or {})

    # 模型特定：预处理
    def preprocess(self, image_bgr, meta_in=None):
        """
        把一张图转成模型输入张量。未实现时抛错，cli 会给出明确提示
        """
        raise NotImplementedError(
            "adapter '%s' 未实现预处理；请改为直接提供 corpus（例如用生产侧 dump）。" % self.name
        )

    # 模型特定：解码
    def decision(self, logits):
        """
        把模型输出张量转成该模型的"结论"（文本/框/类别…）。

        返回可 json 序列化的 dict；generic 适配器返回 None。
        """
        return None

    def decision_digest(self, dec):
        return ""

    # 模型特定：决策层对比（逐 case）
    def decision_compare(self, ref_dec, eng_dec):
        return {}

    # cli 依赖下面这份「标准契约键」（全部可选，缺省视为 0），用于判定逻辑与汇总表：
    #   num_decision_mismatch   结论不一致的样本数
    #   num_fragile_flips       边界脆弱翻转步数
    #   num_structural_flips    结构性翻转步数
    # 除此之外 adapter 可以返回任意扩展键，会原样并入 compare 的 summary。
    def decision_summary(self, rows):
        """
        汇总逐 case 的决策层对比结果，返回 dict（会并入 compare 的 summary）。

        rows: compare 的逐 case 结果列表，每项形如
              {"case_id": str, "numeric": {...}, "decision": <decision_compare 的返回>}
        """
        return {}

    def render_decision_report(self, rows):
        """
        渲染决策层 Markdown 小节，返回行列表（不含结尾空行）。

        rows 同上
        """
        return []

    # 自检
    def validate(self, model_info):
        """
        用模型元信息做自检（如类别数与字典是否配套）。

        model_info: {"input_shapes": [...], "output_shapes": [...], "backend": ...}
        返回警告字符串列表；空列表表示通过。
        """
        return []
