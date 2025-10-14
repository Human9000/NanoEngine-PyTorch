from typing import List, Tuple, Dict
import copy
import torch
from torch import nn
from src.qnn.quantize import *
from torch.fx import Tracer, GraphModule
import onnx
from onnx import shape_inference
import re

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QRule:
    pattern: str  # 正则表达式
    bits: Optional[int] = field(default=None)  # 量化位宽
    lr: Optional[float] = field(default=None)  # 学习率
    c_dim: Optional[int] = field(default=None)  # 通道维度

    def __post_init__(self):
        # 编译一次，后面匹配更快
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def get(self, item, default=None):
        res = getattr(self, item, default)
        if res is None:
            return default
        return res


# 创建一个自定义的 Tracer
class DecomposeTracer(Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # 指定需要分解的模块（里面包含可训练张量的都需要继续分解）
        if m.__module__ in [
            "torch.nn.modules.conv",
            "torch.nn.modules.linear",
            "torch.nn.modules.pooling",
            "torch.nn.modules.upsampling",
        ]:
            return False
        # 对于其他 nn 模块，保持默认行为
        super().is_leaf_module(m, module_qualified_name)


def get_quantization_info(quantization_gm, target, _quantization_rules):
    bits = 8
    lr = 0.1
    scale_shape = None
    c_dim = None  # 默认 channel dim 为 None，即不采用通道级别缩放，只采用tensor级别缩放
    if _quantization_rules is not None:  # 自定义了节点的量化信息
        # 遍历查找当前节点是否在自定义的节点里面
        for rules in _quantization_rules:
            if re.search(rules.pattern, target) is not None:
                bits = rules.get("bits", bits)
                c_dim = rules.get("c_dim", c_dim)
                lr = rules.get("lr", lr)
                if c_dim is not None:
                    param = quantization_gm.get_parameter(target)
                    scale_shape = [1, ] * param.ndim
                    scale_shape[c_dim] = param.shape[c_dim]

    return bits, lr, scale_shape


def graph_module_insert_quantization_nodes(
        quantization_gm: GraphModule,
        customer_rules: Optional[List[QRule]] = None,  # [[r"0\.conv\.conv\d+\.weight", {"bits": 4, "lr": 0.01, "channel_dim": 0}],],  # 用户自定义规则
        default_rules: Optional[List[QRule]] = None,  # [ ,],  # 用户自定义规则
):
    if default_rules is None:
        # 正则表达式自定义可训练参数的量化规则， 若 channel_dim = None, 则采用 tensor scale，否则采用 channel scale
        default_rules = [
            QRule(r".", bits=8, lr=0.1, c_dim=None),  # 默认规则
            QRule(r"\.weight$", bits=8, c_dim=None),  # 默认规则
            QRule(r"\.bias$", bits=8, c_dim=None)  # 默认规则
        ]
    if customer_rules is None:
        customer_rules = []

    # 步骤 1: 准备用于存放 get_attr 量化器的 ModuleDict
    attr_quantize_container_name = "__attr_qs__"
    if not hasattr(quantization_gm, attr_quantize_container_name):
        quantization_gm.add_module(attr_quantize_container_name, nn.ModuleDict())

    attr_quantize_container = getattr(quantization_gm, attr_quantize_container_name)

    for node in list(quantization_gm.graph.nodes):
        if node.op in ['call_module', 'call_function', 'call_method', 'placeholder', 'get_attr']:
            original_users = list(node.users)
            # 如果输出会被量化节点使用，则不再额外添加量化节点
            if len(original_users) > 0 and original_users[0].name.endswith("_q"):
                continue

            quantize_target = f"{node.name}_dq"
            quantize_name = f"{node.name}_q"
            bits, lr, scale_shape = get_quantization_info(quantization_gm,
                                                          str(node.target),
                                                          default_rules + customer_rules)
            quantize = DyQuantize(bits, lr, shape=scale_shape)
            # quantization_gm.add_module(quantize_target, quantize)

            # 步骤 2 & 3: 区分节点类型并定向添加模块
            if node.op == 'get_attr':
                # 将量化器添加到 ModuleDict 中
                attr_quantize_container[quantize_target] = quantize

                # 步骤 4: 调整 call_module 的 target，使其指向 ModuleDict 中的特定 key
                quantize_target = f"{attr_quantize_container_name}.{quantize_target}"
            else:
                # 对于其他类型的节点，行为保持不变
                quantization_gm.add_module(quantize_target, quantize)

            #
            # attr_quantizer_container_name = "get_attr_quantizers"
            # if not hasattr(quantization_gm, attr_quantizer_container_name):
            #     quantization_gm.add_module(attr_quantizer_container_name, nn.ModuleList())
            # # 步骤 2 & 3: 区分节点类型并定向添加模块
            # if node.op == 'get_attr':
            #     # 步骤 4: 调整 call_module 的 target，使其指向 ModuleList 中的特定索引
            #     quantize_target = f"{attr_quantizer_container_name}.{quantize_target}"
            #     # 对于 get_attr 节点，我们将量化器 append 到 ModuleList 中
            #     attr_quantizer_container.append(quantize)
            # else:
            #     quantization_gm.add_module(quantize_target, quantize)

            with quantization_gm.graph.inserting_after(node):
                quantize_node = quantization_gm.graph.call_module(quantize_target, args=(node,))
                quantize_node.name = quantize_name
            for user_node in original_users:
                user_node.replace_input_with(node, quantize_node)

    quantization_gm.recompile()
    return quantization_gm


def remove_quantization_nodes(quantized_gm: GraphModule) -> GraphModule:
    """
    从一个已经插入量化节点的 GraphModule 中移除所有量化节点。
    """
    quantized_gm = copy.deepcopy(quantized_gm)
    # 遍历计算图的所有节点
    for node in list(quantized_gm.graph.nodes):
        # 判断当前节点是否是一个调用 DyQuantize 模块的节点
        if node.op == 'call_module':
            # 通过 node.target 获取模块的名称，再从模型中获取模块实例
            module = getattr(quantized_gm, node.target)
            if isinstance(module, DyQuantize):
                # 获取量化节点的输入节点 (它只有一个参数)
                input_node = node.args[0]
                # 将所有使用该量化节点输出的地方，全部改为直接使用其输入
                # PyTorch FX 提供了非常方便的 API
                node.replace_all_uses_with(input_node)

    # 删除图中不再被使用的节点（即我们刚刚绕过的所有量化节点）
    quantized_gm.graph.eliminate_dead_code()

    # 可选：删除模型中不再被引用的 DyQuantize 模块
    # FX 的 recompile 通常能处理好，但手动清理更彻底
    modules_in_graph = {node.target for node in quantized_gm.graph.nodes if node.op == 'call_module'}
    all_modules = dict(quantized_gm.named_modules())
    for name, module in all_modules.items():
        if isinstance(module, DyQuantize) and name not in modules_in_graph:
            # 在 Python 3.8+ 中，可以用 delattr
            parent_name, _, child_name = name.rpartition('.')
            if parent_name:
                parent_module = quantized_gm.get_submodule(parent_name)
                delattr(parent_module, child_name)
            else:
                delattr(quantized_gm, child_name)

    # 重新编译模块以应用图的更改
    quantized_gm.recompile()

    # 注意：这个模型仍然是 functional 版本的（例如 FuncConv2d）
    # 如果要完全恢复，还需要一个逆向的 replace 过程
    return quantized_gm


def ex_quantize_model_fully_encapsulated(model: GraphModule):
    # 遍历所有的TrainQ 节点 替换为ConstQ结点
    for node in list(model.graph.nodes):
        if node.op == 'call_module' and node.target.endswith('_dq'):
            dq_name = node.target
            cq_name = dq_name[:-2] + "cq"
            dq_module = model.get_submodule(dq_name)
            model.add_module(cq_name, dq_module.to_const())
            node.target = cq_name
            delattr(model, dq_name)

    # 4. 别忘了重新生成代码
    model.graph.lint()
    model.recompile()
    return model


def sanitize_onnx_names(model: onnx.ModelProto) -> Dict[str, str]:
    """
    返回旧名→新名的映射，方便外部反向查找。
    原地修改 model，把所有 . 换成 _，且不以数字开头。
    """
    tbl: Dict[str, str] = {"output": "output"}

    from collections import defaultdict

    # 记录每种 op_type 的计数器
    op_counters = defaultdict(int)
    out_tensor_map = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_counters[op_type] += 1
        new_node_name = f"{op_type}_n{op_counters[op_type]}"
        tbl[node.name] = new_node_name
        # 2. 重命名节点输出 tensor
        for index, out_name in enumerate(node.output):
            out_tensor_map[out_name] = f"{new_node_name}_o{index}"

    # print(out_tensor_map)
    def fix(name: str) -> str:
        if name in tbl:
            return tbl[name]
        if name in out_tensor_map:
            name = out_tensor_map[name]
        # 1. 点 → 下划线
        new = name.replace('.', '_').replace('/', '_')
        # 2. 去连续下划线
        new = re.sub(r'_+', '_', new)
        # 3. 不能以下划线开头
        if new[0] == '_':
            new = new[1:]
        # 3. 不能以数字开头
        if re.match(r'\d', new):
            new = 'Net_' + new
        tbl[name] = new
        return new

    # 1. 节点
    for n in model.graph.node:
        n.name = fix(n.name)
        n.input[:] = [fix(i) for i in n.input]
        n.output[:] = [fix(o) for o in n.output]

    # 2. 张量（权重 / bias）
    for init in model.graph.initializer:
        init.name = fix(init.name)

    # 3. 值信息（中间 tensor）
    for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        v.name = fix(v.name)

    return tbl


def translate_model_to_GraphModule(model: nn.Module) -> GraphModule:
    return GraphModule(model, DecomposeTracer().trace(model))

#
# # --- Example Usage ---
# if __name__ == '__main__':
#     class Conv2(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=0)
#             self.conv2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
#
#         def forward(self, x):
#             return self.conv2(self.conv1(x))
#
#
#     class ComprehensiveModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             # self.conv = nn.Conv1d(3, 16, 3, stride=2, padding=0)
#             self.conv = Conv2()
#             self.custom_param = nn.Parameter(torch.randn(1))
#
#         def forward(self, x):
#             x1 = self.conv(x)
#             x = x1 + self.custom_param + torch.randn(1) + 2
#             x = F.relu(x) + x
#             return x
#
#
#     model = nn.Sequential(ComprehensiveModel())
#
#     print("--- 原始模型 ---")
#     print(model)
#     model.forward(torch.randn(1, 3, 8, 8))
#     # 使用自定义的 DecomposeTracer 展开 model 的 graph 结构
#     quantized_gm = GraphModule(model, DecomposeTracer().trace(model))
#     print("--- 图结构模型 ---")
#     print(quantized_gm)
#     print(list(quantized_gm.state_dict().keys()))
#
#     # 2. Trace the model and insert quantization nodes
#     quantized_gm = graph_module_insert_quantization_nodes(
#         quantized_gm,
#         customer_rules=[
#             QRule(r"conv1\.weight", 4, 0.01, 0),
#             # {"pattern": r"0\.conv\.conv1\.weight", "bits": 4, "lr": 0.01, "channel_dim": 0},  # 自定义规则
#         ]
#     )
#
#     print("\n--- 量化后的模型结构 (完全封装) ---")
#     print(quantized_gm)
#
#     print("\n--- 量化后的模型代码 ---")
#     print(quantized_gm.code)
#
#     print("\n--- 量化后的图结构 (最终的清晰版本!) ---")
#     quantized_gm.graph.print_tabular()
#
#     gm_back = remove_quantization_nodes(quantized_gm)
#     print("\n--- 反量化后的模型结构 (完全封装) ---")
#     print(gm_back)
#
#     print("\n--- 反量化后的模型代码 ---")
#     print(gm_back.code)
#
#     print("\n--- 反量化后的图结构 (最终的清晰版本!) ---")
#     gm_back.graph.print_tabular()
#
#     input_tensor = torch.randn(1, 3, 8, 8)
#     try:
#         quantized_gm.train()
#         output = quantized_gm(input_tensor)
#         print("\n最终模型前向传播成功!")
#         print("输出形状:", output.shape)
#     except Exception as e:
#         import traceback
#
#         print(f"\n前向传播时发生错误: {e}")
#         traceback.print_exc()
#
#     # 导出 onnx
#     try:
#         ex_quantized_gm = ex_quantize_model_fully_encapsulated(quantized_gm.eval())
#         print("\n--- 量化后的导出的模型结构 ---")
#         print(ex_quantized_gm)
#         ex_quantized_gm.graph.print_tabular()
#         torch.onnx.export(ex_quantized_gm,
#                           input_tensor,
#                           "../../onnx_model/quantized_model.onnx",
#                           input_names=["input"],
#                           output_names=["output"],
#                           dynamic_axes={"input": {0: "N"},
#                                         "output": {0: "N"}}
#                           )
#         # 1. 载入模型
#         model = onnx.load("../../onnx_model/quantized_model.onnx")
#         # 2. 清理名字
#         sanitize_onnx_names(model)
#         # 3. 推理所有节点的形状
#         inferred = shape_inference.infer_shapes(model)
#         # 4. 覆盖原模型或另存
#         onnx.save(inferred, '../../onnx_model/quantized_model.onnx')
#         print("\nONNX模型导出成功!")
#     except Exception as e:
#         import traceback
#
#         print(f"\nONNX导出时发生错误: {e}")
#         traceback.print_exc()
