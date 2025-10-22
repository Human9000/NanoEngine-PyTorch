import torch
import ultralytics.nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel


def c2f_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through C2f layer."""
    y = self.cv1(x).chunk(2, 1)
    y = [y[0], y[1]]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def yolo_forward(self, x):
    return self.model(x)


C2f.forward = c2f_forward
# Detect.forward = detect_forward

from torch.fx import symbolic_trace, wrap, Tracer, GraphModule

# 1️⃣ 加载模型并设为评估模式
model = YOLO("yolo11n.pt")
model.eval()


# 2️⃣ 创建一个 FX 友好的代理（Proxy）模块
class YoloV8Proxy(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def forward(self, x):
        return self.yolo_model(x)


# wrap('cat')
class YOLOTracer(Tracer):
    def is_leaf_module(self, m, name):
        # 不展开这些子模块
        if m.__class__.__name__ in ["Concat",
                                    "Detect"]:
            return True
        return super().is_leaf_module(m, name)


def compare_outputs(o, t, atol=1e-5):
    if isinstance(o, (list, tuple)):
        return all(compare_outputs(oi, ti, atol) for oi, ti in zip(o, t))
    return torch.allclose(o, t, atol=atol)


# print(model.model)
# print(type(model.model.model))
# print(model.model.model)
# exit()

# 实例化这个代理模块
# proxy_model = YoloV8Proxy(model.model)
proxy_model = model.model.model
proxy_model.eval()

# print(proxy_model)
# replace_detect_inference(proxy_model)

# 3️⃣ 执行深度追踪
print("--- 开始进行深度符号追踪 ---")
try:
    # 使用标准的 symbolic_trace，它现在知道了如何处理 len 和 list
    # traced_model = YOLOTracer().trace(proxy_model)
    # print(proxy_model)
    # print(isinstance(proxy_model, ultralytics.nn.BaseModel))
    trace_model = torch.nn.Sequential()
    for i in range(len(proxy_model)):
        traced_model = proxy_model[i]
        print(type(traced_model),
              isinstance(traced_model, ultralytics.nn.BaseModel),
              traced_model.i,
              traced_model.type,
              )
        trace_model.append(traced_model)
    # trace_model = GraphModule(proxy_model, YOLOTracer().trace(proxy_model))
    print("✅ 深度追踪成功！模型结构已被展开。")

    # 5️⃣ 验证输出是否一致
    print("\n--- 进行一致性测试 ---")
    example_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        # model.model.model = proxy_model
        original_output = model(example_input)
        model.model.model = trace_model
        traced_output = model(example_input)
    exit(0)
    #
    # original_output = proxy_model(example_input)
    # traced_output = traced_model(example_input)

    if isinstance(original_output, (list, tuple)):
        all_close = compare_outputs(original_output, traced_output)
        # all_close = all(torch.allclose(o, t, atol=1e-5) for o, t in zip(original_output, traced_output))
    else:
        all_close = torch.allclose(original_output, traced_output, atol=1e-5)

    if all_close:
        print("✅ 追踪前后模型输出一致！")
    else:
        print("❌ 追踪前后模型输出不一致！")

except Exception as e:
    import traceback

    print(f"\n❌ 深度追踪失败: {e}")
    traceback.print_exc()
