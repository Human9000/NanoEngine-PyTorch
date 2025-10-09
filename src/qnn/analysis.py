# from qnn import quantize_model,un_quantize_model
from onnx import AttributeProto
from torch import nn
import torch
import onnx
from onnx import numpy_helper
import numpy as np

import json
import pandas as pd

from onnx import shape_inference
from dataclasses import dataclass, field
from typing import List, Any, Dict
from qnn.quantizer import q_scale_k, q_packed_bits


# --- 新增: 定义中间表示 (IR) 的数据结构 ---
@dataclass
class TensorIR:
    name: str  # 张量名称
    data: np.ndarray  # 初始化数据
    dims: List[Any]  # 维度信息 e.g., ['N', 3, 224, 224] or [1, 16, 112, 112]
    shape_type: str

    bits: int  # 量化位宽
    qTBits: int  # tensor 级量化信息，缩放位数
    qCBits: np.ndarray  # channel 级量化信息，缩放位数，若channel_bits=NULL,则表示没有 channel 级别的量化信息

    incremental_memory_len: int  # 缓存的长度
    batch_dims: List[int]  # 批处理维度 ，batch_size or heads or ...
    loop_dims: List[int]  # 循环维度 , height, width, deep or ...
    incremental_dim: int  # 增量维度 time, height, width, deep or ...
    channel_dim: int  # 通道维度 channel

    def __post_init__(self):
        if len(self.dims) == 0:
            self.dims = (1,)
        else:
            self.dims = tuple(self.dims)

    def _normal_str(self):
        return f"""
        TensorIR(
            name={self.name},
            data=array{self.data.shape},
            dims={{{self.dims}}},
            bits={self.bits},
            qTBits={self.qTBits}, 
            qCBits={self.qCBits}, 
            incremental_memory_len={self.incremental_memory_len}, 
            batch_dims={self.batch_dims}, 
            loop_dims={self.loop_dims}, 
            incremental_dim={self.incremental_dim},
            channel_dim={self.channel_dim}
        )"""

    def __str__(self):
        return self.c_str()

    def c_str(self):
        # print(f"let {tensor_name}: Tensor = {{type={map[dtype]}, shape:{shape_type}=[{shape_s}], data=[{data}]}} ;")

        # print(self.data.flatten())

        # 转为量化数据
        # self.data
        # 从onnx中获取对应的量化节点
        # 如果 self.data 是需要init的，则用量化节点量化成量化版本的init数据，然后再导出
        # 如果 self.data 是不需要init的，则直接导出
        # raw_data = self.data.astype("int8")
        # 量化
        raw_data = self.data
        if self.qCBits != None:
            # q_data, mask = q_scale_k(self.data, self.bits, self.qCBits)
            q_data = q_scale_k(self.data, self.bits, self.qCBits)
        else:
            # q_data, mask = q_scale_k(self.data, self.bits, np.array(self.qTBits))
            q_data = q_scale_k(self.data, self.bits, np.array(self.qTBits))

        q_packed_bits(q_data, self.bits)  # 将量化数据打包成 int8
        # 打包数据

        data = ', '.join(f"0x{raw_data[i]:0x}{raw_data[i + 1]:0x}{raw_data[i + 2]:0x}{raw_data[i + 3]:0x}" for i in range(0, len(raw_data) - 3, 4))
        str_dims = ', '.join([str(i) for i in self.dims])
        return f"""
uint32_t {self.name}_data[] = {{{data}}};
Tensor {self.name} = {{ 
        dim={{{str_dims}}},
        nDim={len(self.dims)},
        bits={self.bits},
        qTBits={self.qTBits},
        qCBits={self.qCBits},
        p={self.name}_data,
        mLen={self.incremental_memory_len},
        mIdx=0,
}};"""


@dataclass
class NodeIR:
    name: str  # 节点名称
    op_type: str  # 操作类型
    inputs: List[str]  # 输入张量名称
    outputs: List[str]  # 输出张量名称
    attributes: Dict[str, Any]  # 属性字典


@dataclass
class ModelIR:
    name: str
    # 静态图结构
    init_const_tensors: Dict[str, TensorIR]
    init_static_tensors: Dict[str, TensorIR]
    runtime_tensors: Dict[str, TensorIR]
    init_nodes: Dict[str, NodeIR]
    runtime_nodes: Dict[str, NodeIR]
    inputs: List[str]
    outputs: List[str]
    # 动态分析结果
    # analysis_result: AnalysisResult


def get_attr_value(attr: AttributeProto):
    """
    把 ONNX AttributeProto 转换成 Python 原生对象
    返回单值或列表，取决于类型
    """
    # 单个值类型
    if attr.type == AttributeProto.FLOAT:
        return attr.f, 'float'
    if attr.type == AttributeProto.INT:
        return attr.i, 'float'
    if attr.type == AttributeProto.STRING:
        return attr.s.decode('utf-8'), 'string'
    if attr.type == AttributeProto.TENSOR:
        return attr.t, 'tensor'
    if attr.type == AttributeProto.GRAPH:
        return attr.g, 'graph'

    # 列表类型
    if attr.type == AttributeProto.FLOATS:
        return list(attr.floats), 'floats'
    if attr.type == AttributeProto.INTS:  # ← 你要的判断
        return list(attr.ints), 'ints'
    if attr.type == AttributeProto.STRINGS:
        return [s.decode('utf-8') for s in attr.strings], 'strings'

    # 其他罕见类型按需扩展
    return None


def _get_tensor_info(g, tensor_name: str):
    # 1. 先在 graph.input 里找
    for t in g.input:
        if t.name == tensor_name:
            return t

    # 2. 在 graph.output 里找
    for t in g.output:
        if t.name == tensor_name:
            return t

    # 3. 在 initializer 里找（权重 / 偏置）
    for t in g.initializer:
        if t.name == tensor_name:
            return t

    # 4. 在 value_info 里找（中间张量）
    for t in g.value_info:
        if t.name == tensor_name:
            return t

    return None


def _about_with_input(node, input_about_names):
    for input_name in node.input:
        if input_name in input_about_names:
            return True
    return False


def parse_onnx_file(onnx_file):
    model = onnx.load(onnx_file)
    graph = model.graph

    # 初始化的静态 buf
    init_const_tensor = {init.name: init for init in graph.initializer}
    init_static_tensor = {}  # 提取 与 input 无关的需要生成的 buf
    init_node = {}  # 与 input 无关的 node
    runtime_node = {}  # 与 input 相关的 运行时 node
    runtime_tensor = {i.name: i for i in graph.input}  # 与 input 相关的 运行时 tensor

    for node_i in graph.node:
        if node_i.op_type == "Constant":
            init_const_tensor[node_i.output[0]] = node_i.attribute[0].t

        elif _about_with_input(node_i, runtime_tensor.keys()):
            runtime_node[node_i.name] = node_i
            for name in node_i.output:
                runtime_tensor[name] = _get_tensor_info(graph, name)
        else:
            init_node[node_i.name] = node_i
            for name in node_i.output:
                init_static_tensor[name] = _get_tensor_info(graph, name)
    # exit(0)
    print("init_const_tensor", init_const_tensor.keys())
    print("init_static_tensor", init_static_tensor.keys())
    print("init_node", init_node.keys())
    print("runtime_tensor", runtime_tensor.keys())
    print("runtime_node", runtime_node.keys())

    # exit(0)
    input_name_s = [i.name for i in graph.input]
    output_name_s = [o.name for o in graph.output]
    #
    # graph,
    # init_static_tensor,
    # init_dynamic_tensor,
    # init_node,
    # runtime_tensor,
    # runtime_node,
    # input_name_s,
    # output_name_s

    return (
        graph,
        init_const_tensor,
        init_static_tensor,
        init_node,
        runtime_tensor,
        runtime_node,
        input_name_s,
        output_name_s
    )


def _simulate_node_processing(node_name,
                              input_tensors,
                              output_tensors,
                              tensor_state,
                              node_memory,
                              stride=1,
                              receptive_field_size=1,
                              out_size=1,
                              ):
    # 过滤掉非运行时的输入，非运行时的输入都是已知的
    input_tensors = [name for name in input_tensors if name in tensor_state]

    # 0. 把所有新来的数据放入缓存
    for in_name in input_tensors:
        if in_name not in tensor_state:
            continue
        node_memory[node_name][in_name]['count'] += tensor_state[in_name]['count']
        node_memory[node_name][in_name]['max'] = max(node_memory[node_name][in_name]['max'],
                                                     node_memory[node_name][in_name]['count'])
        node_memory[node_name][in_name]['sum'] += tensor_state[in_name]['count']
        # if tensor_state[in_name]['count'] > 0:
        #     tensor_state[in_name]['use'] += 1
        #     tensor_state[in_name]['use_max'] = max(tensor_state[in_name]['use_max'], tensor_state[in_name]['use'])

    can_fire = True
    while can_fire:
        # 1. 检查所有数据输入是否满足执行一次计算的最低要求
        for in_name in input_tensors:
            if node_memory[node_name][in_name]['count'] < receptive_field_size:
                can_fire = False

        # 2. 如果满足条件，则模拟执行计算（“触发”节点）
        if can_fire:
            # 消耗输入数据
            for in_name in input_tensors:
                node_memory[node_name][in_name]['count'] -= stride

            # 产生输出数据
            for out_name in output_tensors:
                tensor_state[out_name]['count'] += out_size
                tensor_state[out_name]['max'] = max(tensor_state[out_name]['max'],
                                                    tensor_state[out_name]['count'])
                tensor_state[out_name]['sum'] += out_size


def _simulate_conv_node(node, fet_buff, node_memory, state_dim=2, out_size=1):
    """模拟卷积节点的处理函数"""
    attributes = {attr.name: attr for attr in node.attribute}
    kernel_shape = attributes["kernel_shape"].ints[state_dim - 2]
    dilation = attributes["dilations"].ints[state_dim - 2]
    stride = attributes["strides"].ints[state_dim - 2]

    # 对于卷积，执行一次计算所需的输入长度（感受野）由卷积核和空洞率决定
    receptive_field_size = dilation * (kernel_shape - 1) + 1

    _simulate_node_processing(node.name,
                              node_memory[node.name].keys(),
                              node.output,
                              fet_buff,
                              node_memory,
                              stride,
                              receptive_field_size,
                              out_size,
                              )


def _simulate_simple_node(node, fet_buffer, node_memory, state_dim=3, out_size=1):
    """模拟简单节点（如Mul, Add, Constant等）的处理函数，这些节点通常是逐元素操作"""
    # 简单节点的感受野为1，步长也为1
    receptive_field_size = 1
    stride = 1
    _simulate_node_processing(node.name,
                              node_memory[node.name].keys(),
                              node.output,
                              fet_buffer,
                              node_memory,
                              stride,
                              receptive_field_size)


def simulation_run_analysis(input_name_s, output_name_s, runtime_tensor_s, runtime_node, state_dim=2, one_step_len=1):
    # 算子类型到其模拟函数的映射
    NODE_SIMULATORS = {
        "Conv": _simulate_conv_node,
        "MaxPool": _simulate_conv_node,
        "Constant": _simulate_simple_node,
        "Mul": _simulate_simple_node,
        "Aul": _simulate_simple_node,
        "Add": _simulate_simple_node,
        "Q": _simulate_simple_node,
        "Relu": _simulate_simple_node,
        "Shape": _simulate_simple_node,
        "ConstantOfShape": _simulate_simple_node,
        # 可以根据需要添加更多算子的模拟器，例如 Add, MaxPool 等
    }
    # 模拟运行图
    # 为每个 runtime_tensor_s 创建状态（当前长度，最大长度，本轮使用次数，最大使用次数）
    # 为每个 runtime_node 的 runtime_tensor_s 的输入创建一个缓存状态（当前长度，最大长度，本轮使用次数，最大使用次数）
    tensor_states = {tensor_name: {'count': 0, 'max': 0, 'sum': 0} for tensor_name in runtime_tensor_s}
    memory_states = {node_name: {tensor_name: {'count': 0, 'max': 0, 'sum': 0}
                                 for tensor_name in node.input if tensor_name in runtime_tensor_s}
                     for node_name, node in runtime_node.items()}

    epoch = 0
    # one step data flow simulation
    while True:
        epoch += 1
        # 每轮重置全部 tensor_states 的 计数器
        for name in tensor_states:
            tensor_states[name]["count"] = 0

        # 每轮重置 input tensors 的  count 和 max， 模拟input 来了一个数据
        for name in input_name_s:
            tensor_states[name]['count'] = one_step_len
            tensor_states[name]['max'] = max(one_step_len, tensor_states[name]['max'])
            tensor_states[name]['sum'] += one_step_len

        # 模拟节点执行
        for node_name, node in runtime_node.items():
            if node.op_type in NODE_SIMULATORS:
                NODE_SIMULATORS[node.op_type](node, tensor_states, memory_states, state_dim)
            else:
                print(f"未知的节点类型: {node.op_type}")

        # 检查是否所有的 out tensor 都有了输出
        all_output_ready = all(tensor_states[name]['count'] > 0 for name in output_name_s)
        if all_output_ready:
            break
    return tensor_states, memory_states, epoch


def print_state(tensor_states, memory_states, mem_dim, dim_step_len_arr):
    node_names = list(memory_states[mem_dim])
    tensor_names = list(tensor_states[mem_dim])
    dims = list(tensor_states.keys())

    # --- 使用 Pandas 打印结果 ---
    print("\n========== 内存缓冲区峰值占用分析结果 ==========")
    print(f"dim_step_len_arr = {dim_step_len_arr}")
    print(f"memory_dim_index = {mem_dim}")
    # 1. 打印张量输出缓冲区 (Tensor Buffers)
    print(f"\n--- 1. 动态张量输出缓冲区 (Tensor Buffers) ---")
    # print("该部分显示了每个中间张量作为“输出”时，在任意时刻需要缓冲的最大数据单元数。")

    # 准备数据
    out_buff_data = []
    for name in tensor_names:
        map = {"Tensor Name": name}
        for dim in dims:
            state = tensor_states[dim][name]
            map[f"Max-Len-D{dim}"] = state['max']
            map[f"Sum-Len-D{dim}"] = state['sum']
        out_buff_data.append(map)

    # 创建并打印DataFrame
    if out_buff_data:
        df_out = pd.DataFrame(out_buff_data)
        print(df_out.to_string(index=False))
    else:
        print("无有效的输出缓冲区数据。")

    # 2. 打印节点输入缓存 (Node Input Caches)
    print("\n--- 2. 节点输入缓存 (Node Input Caches) ---")
    # print("该部分显示了每个节点为了满足其计算（如卷积的感受野），需要在其“输入端”累积的最大数据单元数。")
    # 准备数据
    in_mem_data = []
    for node_name in node_names:
        for tensor_name in memory_states[mem_dim][node_name]:
            map = {"Node Name": node_name, "Tensor Name": tensor_name, }
            for dim in dims:
                state = memory_states[dim][node_name][tensor_name]
                map[f"Max Len D{dim}"] = state['max']
                map[f"Sum Len D{dim}"] = state['sum']

            in_mem_data.append(map)
    # 创建并打印DataFrame
    if in_mem_data:
        df_in = pd.DataFrame(in_mem_data)
        print(df_in.to_string(index=False))
    else:
        print("无有效的节点输入缓存数据。")

    print("\n====================== 分析结束 ======================")


def make_init_const_tensor(init_const_tensor):  # 初始化时期的常量张量
    print("\n// init const tensor")
    for tensor_name, tensor_info in init_const_tensor.items():
        dtype_list = ['UNDEFINED', 'F32', 'U8', 'I8', 'U16', 'I16', 'I32', 'I64', 'STRING', 'BOOL', 'F16', 'F64', 'U32', 'U64', 'COMPLEX64', 'COMPLEX128', 'BF16']
        map = {"F32": 'little_float_32', "I8": "int8_t"}  # 小端 四字节 float 列表
        dtype = dtype_list[tensor_info.data_type]
        raw_data = tensor_info.raw_data
        # np_array = numpy_helper.to_array(tensor_info)
        shape = tensor_info.dims
        shape_type = ""
        for i in range(len(shape)):
            key = {0: "C", 1: "C"}.get(i, f"D")
            shape_type += key
        shape_s = ', '.join([str(i) for i in shape])
        print(f"let {tensor_name}: Tensor<{map[dtype]}> = {{shape=[{shape_s}], data=0x{raw_data.hex()}}} ;")

        # e.g. conv1d weights: [cout, cin, ksize]
        # tensor_ir = TensorIR(
        #     name=tensor_name,
        #     data=np_array,
        #     dims=shape,
        #     shape_type=shape_type,
        #     bits=8,
        #     qTBits=0,
        #     qCBits=None,
        #     incremental_memory_len=0,
        #     batch_dims=[],
        #     loop_dims=[0],
        #     incremental_dim=2,
        #     channel_dim=1
        # )
        # print(tensor_ir)


def make_init_static_tensor(init_static_tensor):  # 初始化时期的静态张量
    print("\n// init static tensor")

    for name, tensor_info in init_static_tensor.items():
        shape = []
        # print(tensor_info.type.tensor_type)
        for d in tensor_info.type.tensor_type.shape.dim:
            if d.HasField('dim_value'):
                shape.append(int(d.dim_value))
            else:
                raise ValueError("未知的维度类型")

        shape_type = ""
        for i in range(len(shape)):
            key = {0: "C", 1: "C"}.get(i, f"D")
            shape_type += key
        shape_s = ', '.join([str(i) for i in shape])
        print(f"let {name}: Tensor<Any>  = {{shape:{shape_type}=[{shape_s}]}};")


def make_runtime_tensor(runtime_tensor, tensor_state, mem_dim):  # 初始化时期的静态张量
    print("\n// runtime tensor")
    for tensor_name, tensor_info in runtime_tensor.items():
        shape = []
        for d in tensor_info.type.tensor_type.shape.dim:
            if d.HasField('dim_value'):
                shape.append(int(d.dim_value))
            elif d.HasField('dim_param'):
                shape.append(d.dim_param)  # 动态维度
            else:
                raise ValueError("未知的维度类型")
        shape[0] = -1
        for dim in tensor_state:
            shape[dim] = tensor_state[dim][tensor_name]['sum']
        shape[mem_dim] = tensor_state[mem_dim][tensor_name]['max']

        shape_type = ""
        for i in range(len(shape)):
            key = {0: "B", 1: "C", mem_dim: "M"}.get(i, f"D")
            shape_type += key
        shape_s = ', '.join([str(i) for i in shape])
        print(f"let {tensor_name}: Tensor<Any> = {{shape:{shape_type}=[{shape_s}]}};")


def make_runtime_memory(runtime_tensor, memory_state, mem_dim):  # 运行时期的缓存
    print("\n// runtime memory")
    for node_name in memory_state[mem_dim]:
        for tensor_name in memory_state[mem_dim][node_name]:
            tensor_info = runtime_tensor[tensor_name]
            shape = []
            for d in tensor_info.type.tensor_type.shape.dim:
                if d.HasField('dim_value'):
                    shape.append(int(d.dim_value))
                elif d.HasField('dim_param'):
                    shape.append(d.dim_param)  # 动态维度
                else:
                    raise ValueError("未知的维度类型")
            shape[0] = -1
            for dim in memory_state:
                shape[dim] = memory_state[dim][node_name][tensor_name]['sum']
            shape[mem_dim] = memory_state[mem_dim][node_name][tensor_name]['max']
            shape_type = ""
            for i in range(len(shape)):
                key = {0: "B", 1: "C", mem_dim: "M"}.get(i, f"D")
                shape_type += key
            shape_s = ', '.join([str(i) for i in shape])
            print(f"let {node_name}_{tensor_name}: mem Tensor = {{type=Any, shape:{shape_type}=[{shape_s}]}};")


def make_init_node(init_node):  # 初始化时期的节点
    print("\n// init node")
    for node_name, node in init_node.items():
        op_type = node.op_type
        attr = []
        for attribute in node.attribute:
            attr_name = attribute.name
            value, vtype = get_attr_value(attribute)
            attr.append(f"{attr_name}={value}")

        in_tensor_s = ", ".join(node.input)
        out_tensor_s = ", ".join(node.output)
        attr_s = ', '.join(attr)

        print(f"""let {node_name}: init Node = {{
    op_type={op_type},
    attr=[{attr_s}],
    in_s=[{in_tensor_s}],
    out_s=[{out_tensor_s}]
}};""")


def make_runtime_node(runtime_node, runtime_tensor):  # 运行时期的节点
    print("\n// runtime_node")
    for node_name, node in runtime_node.items():
        op_type = node.op_type
        attr = []
        for attribute in node.attribute:
            attr_name = attribute.name
            value, vtype = get_attr_value(attribute)
            if attr_name == "scale_k":
                shape_type = ["N", ] * len(value.dims)
                max_index = np.argmax(value.dims)
                shape_type[max_index] = "C"
                shape_s = ''.join(shape_type)

                raw_data = value.raw_data
                data = []
                for i in range(0, len(raw_data), 4):
                    stri = ['0'] * 8
                    stri[0:2] = f"{raw_data[i]:02X}"
                    if i + 1 >= len(raw_data):
                        break
                    stri[2:4] = f"{raw_data[i + 1]:02X}"
                    if i + 2 >= len(raw_data):
                        break
                    stri[4:6] = f"{raw_data[i + 2]:02X}"
                    if i + 3 >= len(raw_data):
                        break
                    stri[6:8] = f"{raw_data[i + 3]:02X}"
                    data.append("0x" + "".join(stri))
                data = ', '.join(data)
                print(f"let {node_name}_scale_k: Tensor = {{type=Any, shape:{shape_s}={value.dims}, data=[{data}]}};")
                attr.append(f"{attr_name}={node_name}_scale_k")
            else:
                attr.append(f"{attr_name}={value}")
        in_tensor_s = ", ".join(node.input)
        mem_tensor_s = ", ".join(node_name + '_' + tensor_name for tensor_name in node.input if tensor_name in runtime_tensor)
        out_tensor_s = ", ".join(node.output)
        attr_s = ', '.join(attr)
        print(f"""let {node_name}: Node = {{
    op_type={op_type},
    attr=[{attr_s}],
    in_s=[{in_tensor_s}],
    mem_s=[{mem_tensor_s}],
    out_s=[{out_tensor_s}]
}};""")


def make_call_node(init_node, runtime_node):
    # 生成init的调用函数
    print("\n// call init node")
    print("func init(){")
    for node_name in init_node:
        print(f"    {node_name}.do();")
    print("}")
    # 生成init的调用函数
    print("\n// call runtime node")
    print("func runtime(){")
    for node_name in runtime_node:
        print(f"    {node_name}.do();")
    print("}")


def main():
    (
        graph,
        init_const_tensor,
        init_static_tensor,
        init_node,
        runtime_tensor,
        runtime_node,
        input_name_s,
        output_name_s,
    ) = parse_onnx_file("../onnx_model/quantized_model.onnx")

    # 2d tensor
    tensor_states = {}
    memory_states = {}

    mem_dim = 2
    dim_step_len_arr = [(2, 1),
                        (3, 5), ]
    for dim, step_len in dim_step_len_arr:
        tensor_state, memory_state, epoch = simulation_run_analysis(input_name_s, output_name_s, runtime_tensor, runtime_node, dim, step_len)
        tensor_states[dim] = tensor_state
        memory_states[dim] = memory_state
        if epoch != 1 and dim != mem_dim:
            _, _, expect_step_len = simulation_run_analysis(input_name_s, output_name_s, runtime_tensor, runtime_node, dim, 1)
            raise ValueError(f"epoch({epoch}) != 1 and dim({dim}) != mem_dim({mem_dim}), because the step_len({step_len}) too small! expect step_len >={expect_step_len}")

    print_state(tensor_states, memory_states, mem_dim, dim_step_len_arr)

    make_init_const_tensor(init_const_tensor)  # 初始化时期的常量张量
    make_init_static_tensor(init_static_tensor)  # 初始化时期的静态张量

    make_runtime_tensor(runtime_tensor, tensor_states, mem_dim)  # 运行时期的张量
    make_runtime_memory(runtime_tensor, memory_states, mem_dim)  # 运行时期的缓存

    make_init_node(init_node)  # 初始化时期的节点

    make_runtime_node(runtime_node, runtime_tensor)  # 运行时期的节点
    make_call_node(init_node, runtime_node)  # 调用节点


if __name__ == "__main__":
    main()
