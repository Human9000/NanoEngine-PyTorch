from qnn.analysis import (parse_onnx_file,
                          simulation_run_analysis,
                          print_state)

with open("a.c", 'w') as f:
    f.write("#include <stdio.h>")

def print(*args):
    with open("a.c", 'a') as f:
        f.write(" ".join(args))




def make_opt_interface():  # 定义算子接口
    print("""      
typedef struct{
    uint32_t [10] dims; // [Memory{0-1},  Length {0-8}, Channel{1-1}] ， {} 里面表示前面这个特征的数量的取值范围，左闭右闭  
    uint8_t n_dim;  // shape 的有效长度
    uint8_t have_memory=True; // True if have memory dim else False
    uint32_t m_len;
    int8_t * m_p;
    int8_t * data;
}Tensor;

void conv1d(
    Tensor input,
    Tensor weight,
    Tensor bias,
    Tensor output,
    uint8_t dilations,
    uint8_t group,
    uint8_t kernel_shape,
    uint8_t [2] pads,
    uint8_t [2] strides,
)

void memory_push(Tensor * memory, Tensor * input){  
    uint32_t one_mem_size = 1;
    for (int i = 1; i < memory->n_dim; i++){  
        one_mem_size +=  memory->dims[i]; 
    } 
    
    uint32_t memory_idx =  memory->m_p + memory->m_len;
    uint32_t input_idx = 0; 
    
    // 循环队列 copy 数据
    for (;input_idx < input->dims[0];input_idx++, memory_idx++){ // 循环 mem_dim 的次数
        memory_idx %= memory->m_len;
        std::memory_copy(
            memory.data + memory_idx * size,
            input.data + input_idx * size,
            size,
        );  // memory 的 mem_dim 提到最外层做索引 
    } 
}
    
void memory_conv1d(
    Tensor * input,
    Tensor * weight,
    Tensor * bias,
    Tensor * memory,
    Tensor * output,
    uint8_t dilations,
    uint8_t group,
    uint8_t kernel_shape,
    uint8_t [2] pads,
    uint8_t [2] strides,
){
    // 如果输入input 和 memory 的形状对不上，则报错
    for(int i=1; i<input->n_dims;i ++{
        if (input->dims[i] != memory->dims[i]){
            printf("Error: input->dims[i] != memory->dims[i]");
        }
        return;
    }
    memory_push(memory, input, input.shape[memory_dim]); 
    conv1d(memory, weight, bias, output, dilations, group, kernel_shape, pads, strides);
    memory_pop(memory, strides[memory_dim-2]);
};""")


def make_init_const_tensor(init_const_tensor):  # 初始化时期的常量张量
    pass


def make_init_static_tensor(init_static_tensor):  # 初始化时期的静态张量
    pass


def make_runtime_tensor(runtime_tensor, tensor_states, mem_dim):  # 运行时期的张量
    pass


def make_runtime_memory(runtime_tensor, memory_states, mem_dim):  # 运行时期的缓存
    pass


def make_init_node(init_node):  # 初始化时期的节点
    pass


def make_runtime_node(runtime_node, runtime_tensor):  # 运行时期的节点
    pass


def make_call_node(init_node, runtime_node):  # 调用节点
    pass


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
    ) = parse_onnx_file("../qnn3/quantized_model.onnx")

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

    make_opt_interface()  # 定义算子接口
    #
    # make_init_const_tensor(init_const_tensor)  # 初始化时期的常量张量
    # make_init_static_tensor(init_static_tensor)  # 初始化时期的静态张量
    #
    # make_runtime_tensor(runtime_tensor, tensor_states, mem_dim)  # 运行时期的张量
    # make_runtime_memory(runtime_tensor, memory_states, mem_dim)  # 运行时期的缓存
    #
    # make_init_node(init_node)  # 初始化时期的节点
    #
    # make_runtime_node(runtime_node, runtime_tensor)  # 运行时期的节点
    # make_call_node(init_node, runtime_node)  # 调用节点


if __name__ == '__main__':
    main()
