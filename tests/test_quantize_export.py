from src import (
    translate_model_to_GraphModule,
    QRule,
    graph_module_insert_quantization_nodes,
    remove_quantization_nodes,
    ex_quantize_model_fully_encapsulated,
    sanitize_onnx_names,
    shape_inference,
)
from torch import nn
import torch
from torch.nn import functional as F
import onnx

# --- Example Usage ---
if __name__ == '__main__':
    class Conv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)

            self.bn = nn.BatchNorm2d(16)

        def forward(self, x):
            x = self.conv2(self.conv1(x))
            y = self.bn(x)
            return y


    class ComprehensiveModel(nn.Module):
        def __init__(self):
            super().__init__()
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
    model.forward(torch.randn(1, 3, 8, 8))
    # 使用自定义的 DecomposeTracer 展开 model 的 graph 结构
    quantized_gm = translate_model_to_GraphModule(model)
    print("--- 图结构模型 ---")
    print(quantized_gm)
    print(list(quantized_gm.state_dict().keys()))

    # 2. Trace the model and insert quantization nodes
    quantized_gm = graph_module_insert_quantization_nodes(
        quantized_gm,
        customer_rules=[
            QRule(r"conv1\.weight", 4, 0.1, None, False),
            QRule(r"conv2\.weight", 1, 0.1, 0, False),
            # {"pattern": r"0\.conv\.conv1\.weight", "bits_len": 4, "lr": 0.01, "channel_dim": 0},  # 自定义规则
        ],
    )

    print("\n--- 量化后的模型结构 (完全封装) ---")
    print(quantized_gm)

    print("\n--- 量化后的模型代码 ---")
    print(quantized_gm.code)

    print("\n--- 量化后的图结构 (最终的清晰版本!) ---")
    quantized_gm.graph.print_tabular()

    exit(0)

    gm_back = remove_quantization_nodes(quantized_gm)
    print("\n--- 反量化后的模型结构 (完全封装) ---")
    print(gm_back)

    print("\n--- 反量化后的模型代码 ---")
    print(gm_back.code)

    print("\n--- 反量化后的图结构 (最终的清晰版本!) ---")
    gm_back.graph.print_tabular()

    input_tensor = torch.randn(1, 3, 8, 8)
    try:
        quantized_gm.train()
        output = quantized_gm(input_tensor)
        print("\n最终模型前向传播成功!")
        print("输出形状:", output.shape)
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
                          "../onnx_model/quantized_model.onnx",
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "N"},
                                        "output": {0: "N"}}
                          )
        # 1. 载入模型
        model = onnx.load("../onnx_model/quantized_model.onnx")
        # 2. 清理名字
        sanitize_onnx_names(model)
        # 3. 推理所有节点的形状
        inferred = shape_inference.infer_shapes(model)
        # 4. 覆盖原模型或另存
        onnx.save(inferred, '../onnx_model/quantized_model.onnx')
        print("\nONNX模型导出成功!")
    except Exception as e:
        import traceback

        print(f"\nONNX导出时发生错误: {e}")
        traceback.print_exc()
