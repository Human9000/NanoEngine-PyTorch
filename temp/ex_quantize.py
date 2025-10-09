from typing import List, Tuple, Dict
import copy
import torch
from torch import nn
from qnn.quantizer import *
from torch.fx import Tracer, GraphModule
import onnx
from onnx import shape_inference
import re


# --- The Final, Fully Encapsulated Graph Transformation ---
def quantize_model_fully_encapsulated(
        model: nn.Module,
        weight_bits: int,
        acc_bits: int,
        scale_lr: float
) -> GraphModule:
    module_map = {
        nn.Conv1d: FuncConv1d,
        nn.Conv2d: FuncConv2d,
        nn.Conv3d: FuncConv3d,
    }
    # 1. Replace stateful modules with their functional wrappers
    quantized_model = copy.deepcopy(model)
    replacements: List[Tuple[nn.Module, str, nn.Module]] = []
    for name, module in quantized_model.named_modules():
        if type(module) in module_map:
            new_module = module_map[type(module)](module)
            parts = name.split('.')
            if len(parts) == 1 and parts[0] == "":
                continue
            parent = quantized_model
            for i, part in enumerate(parts[:-1]):
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            attr_to_replace = parts[-1]
            replacements.append((parent, attr_to_replace, new_module))
    for parent, attr_name, new_module in replacements:
        if attr_name.isdigit():
            parent[int(attr_name)] = new_module
        else:
            setattr(parent, attr_name, new_module)

    # 2. Trace the model and insert quantization nodes
    tracer = Tracer()
    graph = tracer.trace(quantized_model)
    quantized_gm = GraphModule(quantized_model, graph)
    for node in list(quantized_gm.graph.nodes):
        if node.op in ['call_module', 'call_function', 'call_method', 'placeholder', 'get_attr']:
            original_users = list(node.users)
            quantizer_target = f"{node.name}_dq"
            quantizer_name = f"{node.name}_q"
            bits = acc_bits
            if 'weight' in node.name:
                bits = weight_bits
            quantizer = DyQuantizer(bits, scale_lr, in_channel=)
            quantized_gm.add_module(quantizer_target, quantizer)
            with quantized_gm.graph.inserting_after(node):
                quantizer_node = quantized_gm.graph.call_module(quantizer_target, args=(node,))
                quantizer_node.name = quantizer_name
            for user_node in original_users:
                user_node.replace_input_with(node, quantizer_node)

    quantized_gm.recompile()
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
    tbl: Dict[str, str] = {"output":"output"}

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

    print(out_tensor_map)
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


# --- Example Usage ---
if __name__ == '__main__':

    class Conv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=0)

        def forward(self, x):
            return  self.conv2(self.conv1(x))


    class ComprehensiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            # self.conv = nn.Conv1d(3, 16, 3, stride=2, padding=0)
            self.conv = Conv2()
            self.custom_param = nn.Parameter(torch.randn(1))

        def forward(self, x):
            x1 = self.conv(x)
            x = x1 + self.custom_param + torch.randn(1) + 2
            x = F.relu(x) + x
            return x


    model = nn.Sequential(ComprehensiveModel())

    print("--- 原始模型 ---")
    print(model)

    quantized_gm = quantize_model_fully_encapsulated(model, 1, 8, 0.1)
    quantized_gm.eval()
    print("\n--- 量化后的模型结构 (完全封装) ---")
    print(quantized_gm)

    print("\n--- 量化后的模型代码 ---")
    print(quantized_gm.code)

    print("\n--- 量化后的图结构 (最终的清晰版本!) ---")
    quantized_gm.graph.print_tabular()

    input_tensor = torch.randn(1, 3, 8, 8)
    try:
        quantized_gm.train()
        output = quantized_gm(input_tensor)
        print("\n最终模型前向传播成功!")
        print("输出形状:", output.dims)
    except Exception as e:
        import traceback

        print(f"\n前向传播时发生错误: {e}")
        traceback.print_exc()

    # 导出 onnx
    try:
        ex_quantized_gm = ex_quantize_model_fully_encapsulated(quantized_gm.eval())
        print("\n--- 量化后的导出的模型结构 ---")
        print(ex_quantized_gm)
        ex_quantized_gm.graph.print_tabular()
        torch.onnx.export(ex_quantized_gm,
                          input_tensor,
                          "quantized_model.onnx",
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "N"},
                                        "output": {0: "N"}}
                          )

        # 1. 载入模型
        model = onnx.load('quantized_model.onnx')
        # 2. 清理名字
        sanitize_onnx_names(model)
        # 3. 推理所有节点的形状
        inferred = shape_inference.infer_shapes(model)
        # 4. 覆盖原模型或另存
        onnx.save(inferred, 'quantized_model.onnx')
        print("\nONNX模型导出成功!")
    except Exception as e:
        import traceback

        print(f"\nONNX导出时发生错误: {e}")
        traceback.print_exc()
