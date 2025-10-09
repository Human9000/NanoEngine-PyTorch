from qnn.quantizer import *
from scripts.quantize_export import quantize_model_fully_encapsulated

# --- Example Usage ---
if __name__ == '__main__':
    class Conv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=0)
            self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=0)

        def forward(self, x):
            return self.conv2(self.conv1(x))


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

    # 1. Replace stateful modules with their functional wrappers

    quantized_gm = quantize_model_fully_encapsulated(model, 1, 8, 0.1)

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
        print("输出形状:", output.shape)
    except Exception as e:
        import traceback

        print(f"\n前向传播时发生错误: {e}")
        traceback.print_exc()
