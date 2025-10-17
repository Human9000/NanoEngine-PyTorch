# NanoEngine-Pytorch


## 功能：
1. 8bit 量化  √
2. 8bit 量化通道级缩放因子  √
3. 可学习缩放因子  √
4. bn层支持  √
5. 1bit量化  √
6. 1bit量化支持通道级缩放因子  √
7. 1bit量化支持可学习缩放因子  √
8. 全程量化（不需要反量化层）  √
9. 量化导出  √
10. 量化模型推理
11. 量化模型分析
12. 增量流式维度推理   √


## v1.0.0-alpha.1
Initial commit
## v1.0.0-alpha.2
修复了低bit概率 导致loss不降低的问题 
## v1.0.0-alpha.3 
修复1bit连用的bug，添加非1bit的通道级别量化支持 
## v1.0.0-alpha.4 
提供了bn层的支持，1bit 量化的缩放因子可学习，1bit量化也支持了通道级缩放
## v1.0.0-beta.1
提供了pip安装方式







```shell
python scripts/quantize_export.py
python qnn/analysis.py
```

```shell 
D:\ProgramData\miniconda3\envs\tch2\python.exe D:/2025/NanoEngine/NanoEngine-Pytorch/qnn/analysis.py
init_const_tensor dict_keys(['tensor_constant0', 'Net_0_custom_param', 'Net_0_conv_conv1_weight', 'Net_0_conv_conv1_bias', 'Net_0_conv_conv2_weight', 'Net_0_conv_conv2_bias', 'Constant_n1_o0', 'Constant_n2_o0', 'Constant_n3_o0', 'Constant_n4_o0', 'Constant_n5_o0', 'Constant_n6_o0', 'Constant_n7_o0', 'Constant_n8_o0', 'Constant_n9_o0', 'Constant_n10_o0', 'Constant_n11_o0', 'Constant_n12_o0', 'Constant_n13_o0', 'Constant_n14_o0', 'Constant_n15_o0'])
init_static_tensor dict_keys(['Q_n2_o0', 'Q_n3_o0', 'Q_n5_o0', 'Q_n6_o0', 'Q_n8_o0', 'Q_n10_o0'])
init_node dict_keys(['Q_n2', 'Q_n3', 'Q_n5', 'Q_n6', 'Q_n8', 'Q_n10'])
runtime_tensor dict_keys(['input', 'Q_n1_o0', 'Conv_n1_o0', 'Q_n4_o0', 'Conv_n2_o0', 'Q_n7_o0', 'Add_n1_o0', 'Q_n9_o0', 'Add_n2_o0', 'Q_n11_o0', 'Add_n3_o0', 'Q_n12_o0', 'Relu_n1_o0', 'Q_n13_o0', 'Add_n4_o0', 'output'])
runtime_node dict_keys(['Q_n1', 'Conv_n1', 'Q_n4', 'Conv_n2', 'Q_n7', 'Add_n1', 'Q_n9', 'Add_n2', 'Q_n11', 'Add_n3', 'Q_n12', 'Relu_n1', 'Q_n13', 'Add_n4', 'Q_n14'])

========== 内存缓冲区峰值占用分析结果 ==========
dim_step_len_arr = [(2, 1), (3, 5)]
memory_dim_index = 2

--- 1. 动态张量输出缓冲区 (Tensor Buffers) ---
Tensor Name  Max-Len-D2  Sum-Len-D2  Max-Len-D3  Sum-Len-D3
      input           1           3           5           5
    Q_n1_o0           1           3           5           5
 Conv_n1_o0           1           1           3           3
    Q_n4_o0           1           1           3           3
 Conv_n2_o0           1           1           3           3
    Q_n7_o0           1           1           3           3
  Add_n1_o0           1           1           3           3
    Q_n9_o0           1           1           3           3
  Add_n2_o0           1           1           3           3
   Q_n11_o0           1           1           3           3
  Add_n3_o0           1           1           3           3
   Q_n12_o0           1           1           3           3
 Relu_n1_o0           1           1           3           3
   Q_n13_o0           1           1           3           3
  Add_n4_o0           1           1           3           3
     output           1           1           3           3

--- 2. 节点输入缓存 (Node Input Caches) ---
Node Name Tensor Name  Max Len D2  Sum Len D2  Max Len D3  Sum Len D3
     Q_n1       input           1           3           5           5
  Conv_n1     Q_n1_o0           3           3           5           5
     Q_n4  Conv_n1_o0           1           1           3           3
  Conv_n2     Q_n4_o0           1           1           3           3
     Q_n7  Conv_n2_o0           1           1           3           3
   Add_n1     Q_n7_o0           1           1           3           3
     Q_n9   Add_n1_o0           1           1           3           3
   Add_n2     Q_n9_o0           1           1           3           3
    Q_n11   Add_n2_o0           1           1           3           3
   Add_n3    Q_n11_o0           1           1           3           3
    Q_n12   Add_n3_o0           1           1           3           3
  Relu_n1    Q_n12_o0           1           1           3           3
    Q_n13  Relu_n1_o0           1           1           3           3
   Add_n4    Q_n13_o0           1           1           3           3
   Add_n4    Q_n12_o0           1           1           3           3
    Q_n14   Add_n4_o0           1           1           3           3

====================== 分析结束 ======================

// init const tensor
let tensor_constant0: Tensor<little_float_32> = {shape=[1], data=0x41f3f63f} ;
let Net_0_custom_param: Tensor<little_float_32> = {shape=[1], data=0x78445fbf} ;
let Net_0_conv_conv1_weight: Tensor<little_float_32> = {shape=[16, 3, 3, 3], data=0x20728f3c2d06083eb0e2c73cd9c10c3e6c5d3fbd385a093d79be3e3e5819f0bcc08744bc413bc5bdc09099bcb425433db95f223e283001bec377093ef8d1d9bc261cdcbdf34c2b3eaeb1fb3d1e87cf3d2004133ca01b833b82048f3d9a72b2bdaa1a80bdc08b6ebd84b3733df8c5b03c3c4c753d5af9dc3d6052173df4519cbdb778eabd38e52d3d06b195bd68e21e3d14334abd20bf9fbb8cee16bddc1060bd9768113e4eb3e03d686a80bd6e89983d148e04bd26a7843dc3403dbe7ffe95bda0a5fe3c9ad4a53de874cfbc8901363e223e9f3d7e56fe3d83d6acbd682b7cbde8e197bcbf8e093efa64e13d40ead33cce8afb3d60a9743c39ee0a3e35b0343e0ae423be8cba3cbdf280b3bd3d8e2fbecbb319beb00b12bc9d620e3e22483ebe211b39befc311c3dc84c31bdd0b1dabd103930be415ff7bdb891513d369bd23dc06f84bdaeae12be627afc3dbf99ecbdd6a92dbe703ae03c27ef2cbe62d6993dd85e07be929c15be82e99cbd88dfb43c47e4b0bd0c6c1ebe87e7093e267c8a3d394218bef294a23d184328be9151283ee563083e89ef92bda0f778bdddb43abebdb1a6bd1c092dbd38949bbc1a87e63d6160413e9c3f48bd501b753c3042643c4fa4a7bd77de203ed8f6af3cd2e9a43d3842e7bcae23ef3dd2a137be567b29be8c6eddbd80add83bea4886bd007509bbe0e281bd910c213ed0ce2bbc427f9dbdf73a98bdf3d4f0bd26a7d33debcd413ecf84f2bdfe318a3d4611df3df05bbabcb01e76bc70f6d2bcb48343be6044373cf44e68bd212d2fbe646d2f3ded6a213ea890aa3cdc537e3deeca943d906811bee4df06bee7d5333e0013f73b52efbf3dd04d243c71d98bbd8882eb3cacdc2f3dd7770f3ee21c1ebeb1ae0ebe4f4614befc3742bd002442be305bbabcb85f86bd141606bee12e3fbe0be2163e78809e3c2818013d4f76f8bde1432d3e78f1eebd20be5b3c3d3006be8a17a63d7d8c0f3e008e74bad826923c4c2a68bdf6851ebe6a63a83d982f0abdb6eaec3da004f53cca68903d28f17dbdd4f0d1bda8d166bdecb018be3070e3bdc67c9cbdc5ce0d3ef00473bd004c013b9e13a83d68a106be28afd13c7782cbbda63e83bd0b7d403eb0137ebccad4f73d7cf674bdede88dbd0096d4bba7c73ebefff92abe309a5e3d9eaaa2bd94243fbe50751abe4027c53b56d1ee3d219aa6bd140a56bd3ca71f3d91f8053e4dcababddf261bbefedcd53dfc7eeebdda5c3ebe1c5d6abdd37a0dbe80a9b33bb834a2bceec5fa3dd04b0ebe8b1a25be9668afbdb0125dbc0465bcbd9d8227be832a353ec060c7bc18fb85bc80d955bd09e99cbd524cd23da4b51c3df2b5c63d2286d03d2ed2e23dc088fa3c7acb14bea3b335beae32b5bdcd5935bec89c1dbe1680cb3d027ad0bd806e573d9199163ee6f0afbd002a87bb410b45beadbfcdbd9839863ca91616bed46077bd006bccba748c3fbd517e303e4f163f3eb6ab12beee2982bd6e3da43dec35293dec2283bd0293a93d524faa3da71dbabd86461fbe449834bdb13f343e6c75febd005a9fba4dd2e1bd442a243d7a73fa3dc2e335be1ed4d33d8b673cbe3b28b9bd6edcdd3d3c9952bde0e8f7bbb0724dbd301165bc06e8a8bd72e331bed5751bbee243de3dd2a2dc3d2bdfd6bdb7ad3fbedf341c3ea2c2f2bdc09c73bda6fb843d809cbabdaa6bbc3d42781abeb97a333ef09114bd1f55383e8dc9443e14d2473d5711c0bdec4a15be80b3e9bb29f7b9bd3a37c9bdd049503d1183253eca3926bea20c21be700f28be8af531bee83b053da5e491bddd20113ea1592c3e11efc4bd1c417e3d73222a3eae24d83d74ac753d90cfc5bd72feff3d0df8343eda51af3d31de223e64b825bd901dda3c241d30be4116303edbeee3bd701903bebc8309bed0f6c2bdac3f073d28b000bece4307bee795e4bd290b323e4d5c99bd1d0a403e4b0b103e20a1aa3c628daf3d1318d2bdd6532ebe0e3ce63d3f74aebdf4cc6ebd3ab58c3d8ac9b83da764293ea1e32b3e1033693c50ed23be478719bea17632be2c620fbe46dc34bef7e93a3ece63c43dd09fa7bcf628ea3dbdcecdbd268b92bd3441373dbe5e8f3d7c3130be3b1231be5a41debdfe8e19bef57f383ef034f73c337d303ec0e84f3b00bcbdba8065653c502538bef18420be1ab120be3718fcbdb8c069bd3946433e8032e1ba1088163d445f753d8d922ebec66fb9bda4010dbe48a2113dec641ebd47212b3eda0a18be00be05bc685c1ebdd29da13d85b717beb15d37bede903fbeccde553dfece9a3d422ef53ddd95013ee0b668bcd819183dcb42273e4739fabdd1452fbe2dee1e3e28852dbe0ec41fbed338143e7d77d3bd600b47bd609f573cbe32e73dbeeb3dbe89b32b3eaadc1cbe9f2d3f3e8e5d20bed07c0b3c30f320be9b8e243ed1fe3c3e} ;
let Net_0_conv_conv1_bias: Tensor<little_float_32> = {shape=[16], data=0x7cdb32bd8e6fc63d402f7fbbccf235bd46c0e0bd45ff163e4d7408be697a8dbd1a23febd52b4d83d4af6e73d7c8f1fbe8400c4bde07660bc4031563c6411ffbd} ;
let Net_0_conv_conv2_weight: Tensor<little_float_32> = {shape=[16, 16, 1, 1], data=0x54c5febd9e2c723e5c1028beacca1dbe2c39e13d00bf14bba016d3bc440b3e3e90c5e33c70cec8bd76292c3ef0df3abe34b6fb3d0024ebbba0a1bbbd3e1055beb45e7c3ea8ea143eec7e843d5eba653edc68de3d5eba1e3e98f3773db085dabc46c2123e401762be128c13befe6c18be209f5fbdcadf563e2a67613e00ecc73b783547be446110be7e0f533ed217513e8a6d363ed4a098bd26d432befa0749be4860533d9034d2bd44a6833df62f073e14eddfbdf4bdfe3d62797a3e84472cbe48b0183e28463abe70d37c3d427a5fbec0de31bc7809523dfa274dbe5803493e58ba3c3e280c75bdd8041abde81f4dbe803dc53cc4509bbdd069023e3449233e84cc0f3e8a9754bed06d8fbd0c76b63dd4dc10be8c0fe4bd60263c3eb454d6bd3caa00be54748e3d9e7f28be9c4c9dbdb8fd043ee832d83d487d04be02a266be8802033ea858d93d16e3233eb4d89a3d8850193d00640f3ad4d306be546160be0867b0bdd8b7f13df24a513ea087203eea8258be044b2fbea006373d70898f3c94f9bfbdc882603e424b1c3e50ae4a3d46f45abe500dd33dc654733ec060cd3b00635f3be0aa15be6a08223e32467c3eec06353e2817283e00b6ed3d50b9dfbda67c24be14cadcbd58e9943d54362ebe7c134d3e807dcf3cc878c43dc83b933d1aef203ea0e6afbcd8e479bee04ef0bd28f24a3dcee244be603131be86eb47be10738dbd8058ef3cba42033e685268bed860a1bdc05a00bcf4e347be400e513e644fbf3d8496c5bd64a9143ee6de22be620b0abea409fdbd76c91cbe6802a5bdd0d638be3003193db8af7c3e7e5522bef0ee223e2ee437be204dc1bd0076673c64acdd3db8740ebdbeb5533ed86304be1811163d6c2b84bd20c1ccbd2ecc783ede644cbed8eb2bbe287e3fbde02842bcdcf063be188aef3d5068573e24d3723e90377abe92691b3e60c1e53df2c8353ea28016be300b66bdb6da6abe146b82bd742eeb3dca23783ede7219be105da7bd90220dbd487c8bbda860fc3d4867a0bd42ac3b3ee05379bef4d7eebdc4e7d83df8bd973de855393e40223dbd5c9e713ec83c19bdcca434be9c40453ebc82ab3de03cc43d34a831be20a92d3c801b943cd427f0bdf00b09be40654dbd6c61b2bd548e98bd2c778fbda2b27abea0616c3e7c61e7bd30ea4bbd80a8db3db89073be4066a8bd6809383d948e253ea029db3c2a632ebee8b2b43db0024ebe58e2cf3d7819a33d6a402bbec4e04b3ef8c56cbe42e13a3e98d15e3e30d253bd6025d33da4781ebe6854a3bd78a1403e46f27b3ed40667be2c31273ea8d8e2bde8271fbebe03653ea474c8bde455b4bd28c5283d1e8926be18f4d1bd805c40bb349ecf3db0b0a63c6016acbc80d9573db2e478be60ae2ebe80525a3c568f663ea8c34ebe30045c3d46aa253e90b0c73d500134be} ;
let Net_0_conv_conv2_bias: Tensor<little_float_32> = {shape=[16], data=0x1a4543beaed92a3e22df6a3eae6647be8c8e15be78a3273e10e589bddec557be8048173be84674be6a5807beace111be00cea3ba10188f3c9cb6edbdd81fc4bd} ;
let Constant_n1_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n2_o0: Tensor<int8_t> = {shape=[16, 1, 1, 1], data=0x03030303030303030303030303030303} ;
let Constant_n3_o0: Tensor<int8_t> = {shape=[16], data=0x07070707070707070707070707070707} ;
let Constant_n4_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n5_o0: Tensor<int8_t> = {shape=[16, 1, 1, 1], data=0x00000000000000000000000000000000} ;
let Constant_n6_o0: Tensor<int8_t> = {shape=[16], data=0x07070707070707070707070707070707} ;
let Constant_n7_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n8_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n9_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n10_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n11_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n12_o0: Tensor<little_float_32> = {shape=[], data=0x00000040} ;
let Constant_n13_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n14_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;
let Constant_n15_o0: Tensor<int8_t> = {shape=[1], data=0x07} ;

// init static tensor
let Q_n2_o0: Tensor<Any>  = {shape:CCDD=[16, 3, 3, 3]};
let Q_n3_o0: Tensor<Any>  = {shape:C=[16]};
let Q_n5_o0: Tensor<Any>  = {shape:CCDD=[16, 16, 1, 1]};
let Q_n6_o0: Tensor<Any>  = {shape:C=[16]};
let Q_n8_o0: Tensor<Any>  = {shape:C=[1]};
let Q_n10_o0: Tensor<Any>  = {shape:C=[1]};

// runtime tensor
let input: Tensor<Any> = {shape:BCMD=[-1, 3, 1, 5]};
let Q_n1_o0: Tensor<Any> = {shape:BCMD=[-1, 3, 1, 5]};
let Conv_n1_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Q_n4_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Conv_n2_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Q_n7_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Add_n1_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Q_n9_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Add_n2_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Q_n11_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Add_n3_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Q_n12_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Relu_n1_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Q_n13_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let Add_n4_o0: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};
let output: Tensor<Any> = {shape:BCMD=[-1, 16, 1, 3]};

// runtime memory
let Q_n1_input: mem Tensor = {type=Any, shape:BCMD=[-1, 3, 1, 5]};
let Conv_n1_Q_n1_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 3, 3, 5]};
let Q_n4_Conv_n1_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Conv_n2_Q_n4_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Q_n7_Conv_n2_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Add_n1_Q_n7_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Q_n9_Add_n1_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Add_n2_Q_n9_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Q_n11_Add_n2_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Add_n3_Q_n11_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Q_n12_Add_n3_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Relu_n1_Q_n12_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Q_n13_Relu_n1_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Add_n4_Q_n13_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Add_n4_Q_n12_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};
let Q_n14_Add_n4_o0: mem Tensor = {type=Any, shape:BCMD=[-1, 16, 1, 3]};

// init node
let Q_n2: init Node = {
    op_type=Q,
    attr=[bit_len=4],
    in_s=[Net_0_conv_conv1_weight, Constant_n2_o0],
    out_s=[Q_n2_o0]
};
let Q_n3: init Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Net_0_conv_conv1_bias, Constant_n3_o0],
    out_s=[Q_n3_o0]
};
let Q_n5: init Node = {
    op_type=Q,
    attr=[bit_len=1],
    in_s=[Net_0_conv_conv2_weight, Constant_n5_o0],
    out_s=[Q_n5_o0]
};
let Q_n6: init Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Net_0_conv_conv2_bias, Constant_n6_o0],
    out_s=[Q_n6_o0]
};
let Q_n8: init Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Net_0_custom_param, Constant_n8_o0],
    out_s=[Q_n8_o0]
};
let Q_n10: init Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[tensor_constant0, Constant_n10_o0],
    out_s=[Q_n10_o0]
};

// runtime_node
let Q_n1: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[input, Constant_n1_o0],
    mem_s=[Q_n1_input],
    out_s=[Q_n1_o0]
};
let Conv_n1: Node = {
    op_type=Conv,
    attr=[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]],
    in_s=[Q_n1_o0, Q_n2_o0, Q_n3_o0],
    mem_s=[Conv_n1_Q_n1_o0],
    out_s=[Conv_n1_o0]
};
let Q_n4: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Conv_n1_o0, Constant_n4_o0],
    mem_s=[Q_n4_Conv_n1_o0],
    out_s=[Q_n4_o0]
};
let Conv_n2: Node = {
    op_type=Conv,
    attr=[dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]],
    in_s=[Q_n4_o0, Q_n5_o0, Q_n6_o0],
    mem_s=[Conv_n2_Q_n4_o0],
    out_s=[Conv_n2_o0]
};
let Q_n7: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Conv_n2_o0, Constant_n7_o0],
    mem_s=[Q_n7_Conv_n2_o0],
    out_s=[Q_n7_o0]
};
let Add_n1: Node = {
    op_type=Add,
    attr=[],
    in_s=[Q_n7_o0, Q_n8_o0],
    mem_s=[Add_n1_Q_n7_o0],
    out_s=[Add_n1_o0]
};
let Q_n9: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Add_n1_o0, Constant_n9_o0],
    mem_s=[Q_n9_Add_n1_o0],
    out_s=[Q_n9_o0]
};
let Add_n2: Node = {
    op_type=Add,
    attr=[],
    in_s=[Q_n9_o0, Q_n10_o0],
    mem_s=[Add_n2_Q_n9_o0],
    out_s=[Add_n2_o0]
};
let Q_n11: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Add_n2_o0, Constant_n11_o0],
    mem_s=[Q_n11_Add_n2_o0],
    out_s=[Q_n11_o0]
};
let Add_n3: Node = {
    op_type=Add,
    attr=[],
    in_s=[Q_n11_o0, Constant_n12_o0],
    mem_s=[Add_n3_Q_n11_o0],
    out_s=[Add_n3_o0]
};
let Q_n12: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Add_n3_o0, Constant_n13_o0],
    mem_s=[Q_n12_Add_n3_o0],
    out_s=[Q_n12_o0]
};
let Relu_n1: Node = {
    op_type=Relu,
    attr=[],
    in_s=[Q_n12_o0],
    mem_s=[Relu_n1_Q_n12_o0],
    out_s=[Relu_n1_o0]
};
let Q_n13: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Relu_n1_o0, Constant_n14_o0],
    mem_s=[Q_n13_Relu_n1_o0],
    out_s=[Q_n13_o0]
};
let Add_n4: Node = {
    op_type=Add,
    attr=[],
    in_s=[Q_n13_o0, Q_n12_o0],
    mem_s=[Add_n4_Q_n13_o0, Add_n4_Q_n12_o0],
    out_s=[Add_n4_o0]
};
let Q_n14: Node = {
    op_type=Q,
    attr=[bit_len=8],
    in_s=[Add_n4_o0, Constant_n15_o0],
    mem_s=[Q_n14_Add_n4_o0],
    out_s=[output]
};

// call init node
func init(){
    Q_n2.do();
    Q_n3.do();
    Q_n5.do();
    Q_n6.do();
    Q_n8.do();
    Q_n10.do();
}

// call runtime node
func runtime(){
    Q_n1.do();
    Conv_n1.do();
    Q_n4.do();
    Conv_n2.do();
    Q_n7.do();
    Add_n1.do();
    Q_n9.do();
    Add_n2.do();
    Q_n11.do();
    Add_n3.do();
    Q_n12.do();
    Relu_n1.do();
    Q_n13.do();
    Add_n4.do();
    Q_n14.do();
}

Process finished with exit code 0

```