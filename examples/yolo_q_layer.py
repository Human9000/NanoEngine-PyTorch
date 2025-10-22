import torch
from torch.fx import GraphModule
from torch.nn import functional as F

def __deepcopy__(self, memo):
    new_obj = self.__origin_deepcopy__(memo)
    for p in ['f', 'i', 'c', 'type']:
        v = getattr(self, p, None)
        if v is not None:
            setattr(new_obj, p, v)
    memo[id(self)] = new_obj
    return new_obj

def fsilu_forward(x, inplace):
    return x * F.sigmoid(x)


GraphModule.__origin_deepcopy__ = GraphModule.__deepcopy__
GraphModule.__deepcopy__ = __deepcopy__
F.silu = fsilu_forward

from nept import (
    translate_model_to_GraphModule,
    graph_module_insert_quantization_nodes,
    ex_quantize_model_fully_encapsulated,
    QRule,
)

from ultralytics import YOLO
from ultralytics.nn import BaseModel
from ultralytics.nn.modules.block import (C2f, Conv, Bottleneck)

C2f.origin_forward = C2f.forward
C2f.forward = C2f.forward_split

from torch import nn


def q_conv2d(model: nn.Module):
    # 遍历模型，如果有子模块是nn.Conv2d,则用trace
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv2d, C2f, Conv, Bottleneck)):# or module.__module__ in ["ultralytics.nn.modules.block"]:
        # if isinstance(module, (nn.Conv2d))  or module.__module__ in ["ultralytics.nn.modules.block"]:
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
            # gm.graph.print_tabular()
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
    yolo = YOLO("yolo11l")
    import copy
    # print(yolo)
    # exit(0)
    qyolo = copy.deepcopy(yolo)
    q_conv2d(qyolo.model.model)
    # print(qyolo)
    # exit()

    qyolo = copy.deepcopy(qyolo)
    qyolo.eval()
    for name, module in qyolo.model.model.named_modules():
        if getattr(module, "f", None) is None:
            if isinstance(module, torch.fx.GraphModule):
                print("out-modules", name, module.__dict__.keys())
    print("===" * 10)
    # qyolo 导出 onnx
    # exit(0)
    # exit(0)
    example_input = torch.randn(128, 3, 640, 640).cuda()
    # qyolo.export()
    print("\n--- 进行export测试 ---")
    ex_q_conv2d(qyolo).export(format='onnx',
                              opset=11,
                              dynamic=False,
                              # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                              )
    exit(0)
    yolo.export(format='onnx',
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
