from nept import (
    translate_model_to_GraphModule,
    graph_module_insert_quantization_nodes,
    ex_quantize_model_fully_encapsulated,
    QRule,
)
from torch import nn

from ultralytics import YOLO
from ultralytics.nn import BaseModel
from ultralytics.nn.modules.block import (
    Conv,
)
import torch

map_q_module = {

}


def __deepcopy__(self, memo):
    new_obj = self.__origin_deepcopy__(memo)
    for p in ['f', 'i', 'c', 'type']:
        v = getattr(self, p, None)
        if v is not None:
            setattr(new_obj, p, v)
    memo[id(self)] = new_obj
    return new_obj


def replace_fict_copy(obj):
    obj.__origin_deepcopy__ = obj.__deepcopy__
    obj.__deepcopy__ = __deepcopy__.__get__(obj, obj.__class__)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def replace_silu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.SiLU):
            setattr(model, name, SiLU())
        else:
            replace_silu(module)


def q_conv2d(model: nn.Module):
    # 遍历模型，如果有子模块是nn.Conv2d,则用trace
    for name, module in model.named_children():
        if isinstance(module, (Conv, torch.nn.Conv2d)):
            # if isinstance(module, Conv):
            replace_silu(module) # 把 Conv里面的所有nn.SiLU() 替换成 展开的SiLU

            gm = translate_model_to_GraphModule(module)
            gm = graph_module_insert_quantization_nodes(
                gm,
                [
                    QRule("x$", 32, 0.1, 0, False),
                    QRule(".weight$", 1, 0.1, 0, False),
                    QRule(".running_mean$", 8, 0.1, 0, False),
                    QRule(".running_var$", 1, 0.1, 0, False),
                ]
            )
            # gm = ex_quantize_model_fully_encapsulated(gm)
            if getattr(module, "f", None) is not None:
                setattr(gm, "f", module.f)
            else:
                setattr(gm, "f", -1)

            if getattr(module, "i", None) is not None:
                setattr(gm, "i", module.i)
            else:
                setattr(gm, "i", -1)

            setattr(gm, "type", module.type)
            if getattr(module, "c", None) is not None:
                setattr(gm, "c", module.c)
            # print(gm.f)
            replace_fict_copy(gm)
            setattr(model, name, gm)
        else:
            q_conv2d(module)


def ex_q_conv2d(model: nn.Module):
    # 遍历模型，如果有子模块是nn.Conv2d,则用trace
    for name, module in model.named_children():
        print(module.__class__)
        if isinstance(module, (torch.fx.GraphModule)):
            module = ex_quantize_model_fully_encapsulated(module)
            setattr(model, name, module)
        else:
            ex_q_conv2d(module)
    return model


if __name__ == "__main__":
    yolo = YOLO("yolo11n")
    import copy

    qyolo = copy.deepcopy(yolo)
    q_conv2d(qyolo.model.model)
    # print(qyolo)
    # exit()

    qyolo = copy.deepcopy(qyolo)
    qyolo.eval()
    # exit()
    for name, module in qyolo.model.model.named_modules():
        if getattr(module, "f", None) is None:
            if isinstance(module, torch.fx.GraphModule):
                print("out-modules", name, module.__dict__.keys())
    print("===" * 10)
    # qyolo 导出 onnx
    # exit(0)
    # exit(0)
    example_input = torch.randn(128, 3, 640, 640).cuda()

    print("\n--- 进行export测试 ---")
    ex_q_conv2d(qyolo).export(format='onnx',
                              opset=11,
                              dynamic=False,
                              # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                              )

    print("\n--- 进行一致性测试 ---")
    with torch.no_grad():
        yolo = yolo.cuda()
        yolo.eval()
        for i in range(2):
            original_output = yolo(example_input)

        qyolo = qyolo.cuda()
        qyolo.eval()
        for i in range(5):
            traced_output = qyolo(example_input)
