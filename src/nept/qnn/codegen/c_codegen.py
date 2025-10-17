
def make_init_const_tensor(init_const_tensor):  # 初始化时期的常量张量
    print("\n// init const tensor")
    for tensor_name, tensor_info in init_const_tensor.items():
        dtype_list = ['UNDEFINED', 'F32', 'U8', 'I8', 'U16', 'I16', 'I32', 'I64', 'STRING', 'BOOL', 'F16', 'F64', 'U32', 'U64', 'COMPLEX64', 'COMPLEX128', 'BF16']
        map = {"F32": '<little_float_32>'}  # 小端 四字节 float 列表
        dtype = dtype_list[tensor_info.data_type]
        raw_data = tensor_info.raw_data
        data = ', '.join(f"0x{raw_data[i]:02X}{raw_data[i + 1]:02X}{raw_data[i + 2]:02X}{raw_data[i + 3]:02X}" for i in range(0, len(raw_data), 4))
        shape = tensor_info.dims

        shape_type = ""
        for i in range(len(shape)):
            key = {0: "C", 1: "C"}.get(i, f"D")
            shape_type += key
        shape_s = ', '.join([str(i) for i in shape])
        print(f"let {tensor_name}: Tensor = {{type={map[dtype]}, shape:{shape_type}=[{shape_s}], data=[{data}]}} ;")


def make_init_static_tensor(init_static_tensor):  # 初始化时期的静态张量
    print("\n// init static tensor")

    for name, tensor_info in init_static_tensor.items():
        shape = []
        for d in tensor_info.type.tensor_type.dims.dim:
            if d.HasField('dim_value'):
                shape.append(int(d.dim_value))
            else:
                raise ValueError("未知的维度类型")

        shape_type = ""
        for i in range(len(shape)):
            key = {0: "C", 1: "C"}.get(i, f"D")
            shape_type += key
        shape_s = ', '.join([str(i) for i in shape])
        print(f"let {name}: Tensor = {{type=Any, shape:{shape_type}=[{shape_s}]}};")


def make_runtime_tensor(runtime_tensor, tensor_state, mem_dim):  # 初始化时期的静态张量
    print("\n// runtime tensor")
    for tensor_name, tensor_info in runtime_tensor.items():
        shape = []
        for d in tensor_info.type.tensor_type.dims.dim:
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
        print(f"let {tensor_name}: Tensor = {{type=Any, shape:{shape_type}=[{shape_s}]}};")


def make_runtime_memory(runtime_tensor, memory_state, mem_dim):  # 运行时期的缓存
    print("\n// runtime memory")
    for node_name in memory_state[mem_dim]:
        for tensor_name in memory_state[mem_dim][node_name]:
            tensor_info = runtime_tensor[tensor_name]
            shape = []
            for d in tensor_info.type.tensor_type.dims.dim:
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
            attr.append(f"{attr_name}={value}")
        in_tensor_s = ", ".join(node.input)
        mem_tensor_s = ", ".join(node_name + '_' + tensor_name for tensor_name in node.input if tensor_name in runtime_tensor)
        out_tensor_s = ", ".join(node.output)
        attr_s = ', '.join(attr)
        print(f"""let {node_name} Node = {{
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

def make_opt_interface():
    print("""// 算子接口""")
    print("""fn Conv2d(
    input:*Tensor, 
    weight:*Tensor, 
    bias:*Tensor ,
    memory:*Tensor,
    output:*Tensor,
    dilations:u8[2], 
    group=u8,
    kernel_shape=u8[2], 
    pads=u8[4],
    strides=u8[2],
);""")
