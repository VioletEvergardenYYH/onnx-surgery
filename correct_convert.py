#!/usr/bin/env python
# encoding: utf-8
#     Author : wangzw
#     Created Time: Tue 18 Aug 2020 03:41:07 PM CST
#     Last Change : Wed 19 Aug 2020 09:46:08 PM CST

import onnx
import numpy as np
import uuid
from onnx.helper import make_attribute, make_node
from onnx import TensorProto
from onnx import optimizer
import collections
import pdb
import warnings


def get_node_topology(graph, sub_graph=False) : 
    #返回所有可计算节点 index 的拓扑排序
    if sub_graph:
        return len(graph.node)

    node_num = len(graph.node)
    compute_tensor_name_list = set()#input名称集合
    for input in graph.input :
        compute_tensor_name_list.add(input.name)
    topology_list = []
    print('total node num : %d' % node_num)

    initializer_name_set = set() #权重名称集合
    for init in graph.initializer :
        initializer_name_set.add(init.name)
    while len(topology_list) < node_num :
        #print(len(topology_list))
        for node_index in range(node_num) :
            if node_index in topology_list :
                continue
            node = graph.node[node_index]
            can_compute = True
            for input in node.input :
                if input == '' :
                    continue
                elif input in initializer_name_set :
                    continue
                elif input not in compute_tensor_name_list :
                    can_compute = False
                    break
            if can_compute :
                topology_list.append(node_index)

                for output in node.output :
                    compute_tensor_name_list.add(output)
                break
            else :
                # if len(topology_list) == 46:
                #     pdb.set_trace()
                continue
    
    return topology_list
def append_initializer(graph, init) :
    index = len(graph.initializer) - 1
    graph.initializer.insert(index, init)
def node_eq(node1, node2) :
    if node1.op_type != node2.op_type :
        return False
    if node1.input != node2.input :
        return False
    if node1.output != node2.output :
        return False
    return True
def remove_node_by_name(graph, name):
    for node in graph.node:
        if node.name == name:
            graph.node.remove(node)
def remove_node(graph, node) :
    graph.node.remove(node)
def remove_node_list(graph, node_list) :
    for node in node_list :
        remove_node(graph, node)
    print('remove ', len(node_list), ' nodes')
def get_initializer(graph, name) :
    for init in graph.initializer :
        if init.name == name :
            return init
    return None
def remove_initializer(graph, init) :
    print('remove initializer : %s' % init.name)
    graph.initializer.remove(init)
    return 0
def remove_initializer_by_name(graph, init_name) :
    init = get_initializer(graph, init_name)
    if init is None :
        return -1
    return remove_initializer(graph, init)
def is_initializer(graph, name) :
    return get_initializer(graph, name) != None
def get_node_list_from_input_name(graph, input_name) :
    ret_node_list = []
    for node in graph.node :
        if input_name in node.input :
            ret_node_list.append(node)
    return ret_node_list
def get_node_list_from_output_name(graph, output_name) :
    ret_node_list = []
    for node in graph.node :
        if output_name in node.output :
            ret_node_list.append(node)
    return ret_node_list
def get_node_count_from_input_name(graph, input_name) :
    return len(get_node_list_from_input_name(graph, input_name))
def get_node_from_input_name_and_op_type(graph, input_name, op_type) :
    ret_node_list = get_node_list_from_input_name(graph, input_name)
    for node in ret_node_list :
        if node.op_type == op_type :
            return node
    return None
def get_node_list_from_input_name_and_op_type(graph, input_name, op_type) :
    ret_node_list = get_node_list_from_input_name(graph, input_name)
    res = []
    for node in ret_node_list :
        if node.op_type == op_type :
            res.append(node)
    return res
def get_input_output_index(input_or_output, name) :
    for i in range(len(input_or_output)) :
        if input_or_output[i] == name :
            return i
    return -1
def get_node_from_output_name(graph, output_name) :
    for node in graph.node :
        if output_name in node.output :
            return node
    return None
def get_node_from_output_name_and_op_type(graph, output_name, op_type) :
    ret_node_list = get_node_list_from_output_name(graph, output_name)
    for node in ret_node_list :
        if node.op_type == op_type :
            return node
    return None
def get_node_from_intput_name(graph, input_name) :
    for node in graph.node :
        if input_name in node.input :
            return node
    return None
def TensorProtoToNumpy(tensor) :
    data_type = tensor.data_type
    shape = tensor.dims
    data = None
    if data_type == onnx.TensorProto.FLOAT :
        data = np.frombuffer(tensor.raw_data, dtype = np.float32).reshape(shape)
    elif data_type == onnx.TensorProto.INT32 :
        data = np.frombuffer(tensor.raw_data, dtype = np.int32).reshape(shape)
    elif data_type == onnx.TensorProto.INT64 :
        data = np.frombuffer(tensor.raw_data, dtype = np.int64).reshape(shape)
    else :
        print('TensorProtoToNumpy unsupport data type : %d' % data_type)
    return data
def get_initializer_numpy_value(graph, name) :
    init_tensor = get_initializer(graph, name)
    if init_tensor is None :
        return None
    return TensorProtoToNumpy(init_tensor)
def get_attribute(node, name) :
    for attr in node.attribute :
        if attr.name == name :
            return attr
    return None
def get_attribute_f_value(node, name) :
    ret_attr = get_attribute(node, name)
    if ret_attr != None and ret_attr.type == onnx.AttributeProto.FLOAT :
        return ret_attr.f
    return None
def get_attribute_floats_value(node, name) :
    ret_attr = get_attribute(node, name)
    if ret_attr != None and ret_attr.type == onnx.AttributeProto.FLOATS :
        return ret_attr.floats
    return None
def get_attribute_i_value(node, name) :
    ret_attr = get_attribute(node, name)
    if ret_attr != None and ret_attr.type == onnx.AttributeProto.INT :
        return ret_attr.i
    return None
def get_attribute_ints_value(node, name) :
    ret_attr = get_attribute(node, name)
    if ret_attr != None and ret_attr.type == onnx.AttributeProto.INTS :
        return ret_attr.ints
def get_attribute_s_value(node, name) :
    ret_attr = get_attribute(node, name)
    if ret_attr != None and ret_attr.type == onnx.AttributeProto.STRING :
        return ret_attr.s
    return None
def get_node_op_type(node) :
    return node.op_type
def get_node_index(graph, node) :
    for i in range(len(graph.node)) :
        n = graph.node[i]
        if node_eq(node, n) :
            return i
    return None
    


def Tranformer_GELU_fusion(graph) :
    gelu_index = 0
    while True :
        modify = False
        for node_index in get_node_topology(graph) :
            nodes_to_remove = []
            pow_node = graph.node[node_index]
            if pow_node.op_type != 'Pow' :
                continue

            nodes_to_remove.append(pow_node)
            mul_node_1 = get_node_from_input_name_and_op_type(graph, pow_node.output[0], 'Mul')
            if mul_node_1 is not None :
                nodes_to_remove.append(mul_node_1)
            else :
                print ('expect mul_node_1')
                continue

            add_node_1 = get_node_from_input_name_and_op_type(graph, mul_node_1.output[0], 'Add')
            if add_node_1 is not None :
                nodes_to_remove.append(add_node_1)
            else :
                print ('expect add_node')
                continue

            mul_node_2 = get_node_from_input_name_and_op_type(graph, add_node_1.output[0], 'Mul')
            if mul_node_2 is not None :
                nodes_to_remove.append(mul_node_2)
            else :
                print ('expect mul_node_2')
                continue

            tanh_node = get_node_from_input_name_and_op_type(graph, mul_node_2.output[0], 'Tanh')
            if tanh_node is not None :
                nodes_to_remove.append(tanh_node)
            else :
                print ('expect tanh_node')
                continue

            add_node_2 = get_node_from_input_name_and_op_type(graph, tanh_node.output[0], 'Add')
            if add_node_2 is not None :
                nodes_to_remove.append(add_node_2)
            else :
                print ('expect add_node_2')
                continue

            mul_node_3 = get_node_from_input_name_and_op_type(graph, add_node_2.output[0], 'Mul')
            if mul_node_3 is not None :
                nodes_to_remove.append(mul_node_3)
            else :
                print ('expect mul_node_3')
                continue

            mul_node_4 = get_node_from_input_name_and_op_type(graph, mul_node_3.output[0], 'Mul')
            if mul_node_4 is not None :
                nodes_to_remove.append(mul_node_4)
            else :
                print ('expect mul_node_4')
                continue

            modify = True
            gelu_index += 1
            gelu_node_name = 'bert_model/bert/encoder/TransformerEncoder/gelu_node'+str(gelu_index)
            gelu_node = onnx.helper.make_node('Gelu',
                                            name=gelu_node_name,
                                            inputs=[pow_node.input[0]],
                                            outputs=[mul_node_4.output[0]],
            )
            graph.node.insert(node_index, gelu_node)
            print('set gelu node:', gelu_index)


            remove_node_list(graph, nodes_to_remove)
            break


        if not modify :
            break



def bfs_remove(graph, start_node, end_node, connect = False) :
    """
    remove nodes from start_node to end_node with bfs, removed nodes include those two nodes

    start_node: name of the start_node
    end_node: name of the end_node
    connect: if true, last_node(the node after end_node).input[0] = start_node.input[0]
    """
    nodes_to_remove = []
    removed_node = set()
    queue = collections.deque()
    x = ""
    for node in graph.node :
        if node.name == start_node :
            x = node.input[0]
            nodes_to_remove.append(node)
            queue.append(node)
            removed_node.add(node.name)
            break
    if connect == True:
        for node in graph.node:
            if node.name == end_node:
                last_node = get_node_from_intput_name(graph, node.output[0])
                if last_node is not None:
                    last_node.input[0] = x
    while queue :
        node = queue.popleft()
        if node.name == end_node :
            continue
        i = 0
        if node.op_type == 'Loop':
            i = 1
        for next_node in get_node_list_from_input_name(graph, node.output[i]) :
            if next_node.name not in removed_node :
                queue.append(next_node)
                removed_node.add(next_node.name)
                nodes_to_remove.append(next_node)
    
    print('bfs remove total', len(nodes_to_remove), 'nodes')
    remove_node_list(graph, nodes_to_remove)



def FF_optimize(graph) :
    #优化feed forward部分
    
    visited_node = set()
    not_remove_op = ['MatMul', 'Add', 'Gelu', 'Relu', 'Tanh']
    while True :
        modify = False
        queue = collections.deque()
        for node_index in get_node_topology(graph) :
            nodes_to_remove = []
            ln_node = graph.node[node_index]
            if ln_node.op_type != 'LayerNormalization' or get_node_from_input_name_and_op_type(graph, ln_node.output[0], 'Reshape') is None: 
                continue
            x = ln_node.output[0] #原始保存输入
            y = ''
            for node in get_node_list_from_input_name(graph, x) :
                visited_node.add(node.name)
                queue.append(node)
                if node.op_type not in not_remove_op:
                    nodes_to_remove.append(node)

            activate_output = ''
            last_matmul = None
            last_bias = None
            while queue :
                node = queue.popleft()
                Add_node = get_node_from_input_name_and_op_type(graph, node.output[0], 'Add')
                if Add_node is not None and get_node_from_input_name_and_op_type(graph, Add_node.output[0], 'LayerNormalization') is not None:
                    # the last node
                    y = node.output[0]
                    continue
                elif get_node_from_input_name_and_op_type(graph, node.output[0], 'LayerNormalization') is not None:
                    continue
                elif node.op_type == 'MatMul':
                    reshape_node = get_node_from_input_name_and_op_type(graph, node.output[0], 'Reshape')
                    if reshape_node is None:
                        print("no reshape_node after matmul")
                        continue
                    bias_node = get_node_from_input_name_and_op_type(graph, reshape_node.output[0], 'Add')
                    if bias_node is None:
                        print("no bias_node after reshape")
                        continue
                    bias_node.input[0] = node.output[0]

                    activate_node = get_node_from_input_name_and_op_type(graph, bias_node.output[0], 'Gelu')
                    if activate_node is None:
                        activate_node = get_node_from_input_name_and_op_type(graph, bias_node.output[0], 'Relu')
                    if  activate_node is None:
                        last_matmul = node
                        last_bias = bias_node
                    else:
                        node.input[0] = x
                        activate_output = activate_node.output[0]

                for next_node in get_node_list_from_input_name(graph, node.output[0]) :
                    if next_node.name not in visited_node :
                        queue.append(next_node)
                        visited_node.add(next_node.name)
                        if next_node.op_type not in not_remove_op:
                            nodes_to_remove.append(next_node)

            if last_matmul is not None and last_bias is not None:
                last_bias.output[0] = y
                last_matmul.input[0] = activate_output

            
            if len(nodes_to_remove) :
                modify = True
                remove_node_list(graph, nodes_to_remove)
                break
       
        if not modify : #遍历整个计算图的循环如果不是因为修改LN跳出，那就是因为遍历结束了，完成任务跳出死循环
            break
    # bfs_remove(graph, 'bert_model/bert/encoder/Reshape/shape_Concat__20', 'bert_model/bert/encoder/TransformerEncoder/layer_2/attention/self/MultiHeadsDotProductAttentionLayer/mul_1')
    # bfs_remove(graph, 'bert_model/bert/encoder/ones/packed_Concat__15', 'bert_model/bert/encoder/ones')

    




def recognise_node(name) :
    name_list = name.split('/')
    q_list = ['q', 'Q', 'query', 'Query', 'QUERY']
    k_list = ['k', 'K', 'key', 'Key', 'KEY']
    v_list = ['v', 'V', 'value', 'Value', 'VALUE']
    out_list = ['output_transform', 'output']
    find = False
    ret = 0
    for n in name_list :
        if n in q_list:
            if find:
                print('can not recognise because node name is contained by more than one list')
            find = True
            ret = 1
        elif n in k_list :
            if find:
                print('can not recognise because node name is contained by more than one list')
            find = True
            ret = 2
        elif n in v_list :
            if find:
                print('can not recognise because node name is contained by more than one list')
            find = True
            ret = 3
        elif n in out_list :
            if find:
                print('can not recognise because node name is contained by more than one list')
            find = True
            ret = 4
    return ret


def reverse_bfs(graph, start) :
    visited_node = set()
    queue = collections.deque()
    start_list = get_node_list_from_output_name(graph, start)
    for node in start_list :
        queue.append(node)
        visited_node.add(node.name)

    while queue :
        node = queue.popleft()
        if node.op_type == 'Less' or node.op_type == 'Equal':
            return node
        next_list = []
        for i in range(len(node.input)):
            next_list.extend(get_node_list_from_output_name(graph, node.input[i]))

        for next_node in next_list :
            if next_node.name not in visited_node :
                queue.append(next_node)
                visited_node.add(next_node.name)
    return None

def Is_multihead_attention_start(graph, start_node) :
    visited_node = set()
    queue = collections.deque()
    start_list = get_node_list_from_input_name(graph, start_node.output[0])
    for node in start_list :
        queue.append(node)
        visited_node.add(node.name)

    while queue :
        node = queue.popleft()
        if node.op_type == 'Add':
            continue
        elif node.op_type == 'MatMul' and recognise_node(node.name) in [1,2,3]:
            return True
        for next_node in get_node_list_from_input_name(graph, node.output[0]) :
            if next_node.name not in visited_node :
                queue.append(next_node)
                visited_node.add(next_node.name)
    return False



def Multihead_attention_fusion(graph, mother_graph = None) :
    fuse_subgraph = True
    if mother_graph is None:
        mother_graph = graph
        fuse_subgraph = False
    Multihead_attention_op_index = 0
    removed_node = set()

    #step1: make Multihead_attention's mask
    #key_padding_mask_2d: Multihead_attention needs the mask input like: [f,f,f,f,f,t,t,t], where t(True) is the padding token
    #key_padding_mask_4d: this mask directly add to the input of softmax
    key_padding_mask_2d =  '' 
    key_padding_mask_4d =  '' 
    # for node in graph.node:
    #     if node.op_type == 'Softmax' and Is_attn_softmax(graph, node):
    #         mask_input = Is_attn_softmax(graph, node)
    #         mask_node = reverse_bfs(graph, mask_input)
    #         break
    # if mask_node is None:
    #     raise Exception('can not find mask node(node like Equal or Less)')
    # mask_node_index = get_node_index(graph, mask_node)
    # if mask_node.op_type == 'Equal':
    #     cast_node = onnx.helper.make_node('Cast'
    #                                             , name = 'bert_model/bert/encoder/TransformerEncoder/key_mask_cast' 
    #                                             , inputs = [mask_node.output[0]]
    #                                             , outputs = [input_mask]
    #                                             )
    #     cast_node.attribute.insert(0, onnx.helper.make_attribute('to', 7, 'to'))
    #     graph.node.insert(mask_node_index + 1, cast_node)
    # elif mask_node.op_type == 'Less':
    #     not_node = onnx.helper.make_node('Not'
    #                                                 , name = 'bert_model/bert/encoder/TransformerEncoder/key_mask_not' 
    #                                                 , inputs = [mask_node.output[0]]
    #                                                 , outputs = [input_mask+'_before_cast']
    #                                                 )

            

    #     cast_node = onnx.helper.make_node('Cast'
    #                                             , name = 'bert_model/bert/encoder/TransformerEncoder/key_mask_cast' 
    #                                             , inputs = [input_mask+'_before_cast']
    #                                             , outputs = [input_mask]
    #                                             )
    #     cast_node.attribute.insert(0, onnx.helper.make_attribute('to', 7, 'to'))
    #     graph.node.insert(mask_node_index+1, not_node)
    #     graph.node.insert(mask_node_index+2, cast_node)

    #step2: find the last LN node in advance so that can insert output transpose node
    last_ln_node = ''
    max_ln_id = -1
    for node in graph.node:
        if node.op_type == 'LayerNormalization':
            if int(node.name.split('_')[-1]) > max_ln_id:
                max_ln_id = int(node.name.split('_')[-1])
                last_ln_node = node.name
    print('total ',max_ln_id, ' LN nodes')
    print('last ln node:', last_ln_node)
    


    #step3: BFS the MHA graph and get all the initialiser
    #the input usually is B,T,E, but the MHA needs T,B,E so we need two transpose nodes(in & out)
    first_trans_set = False
    last_trans_set = False
    while True :
        modify = False
        queue = collections.deque()
        for node_index in get_node_topology(graph, sub_graph=fuse_subgraph) :
            nodes_to_remove = []
            start_node = graph.node[node_index]
            if start_node.op_type != 'LayerNormalization':
                continue
            if not first_trans_set:
                print('insert trans_in node')
                in_node_list = get_node_list_from_output_name(graph, start_node.input[0])
                if len(in_node_list) :
                    for n in in_node_list:
                        n.output[0] = 'Transpose_in:in'
                else:
                    raise Exception('in_node_list is empty')

                first_trans_set = True
                transpose_in_node = onnx.helper.make_node('Transpose'
                                    , name = 'Transpose_in'
                                    , inputs = ['Transpose_in:in']
                                    , outputs = [start_node.input[0]]
                                    )
                transpose_in_node.attribute.insert(0, onnx.helper.make_attribute('perm', [1,0,2]))
                graph.node.insert(node_index, transpose_in_node)
                continue
            
            if start_node.name == last_ln_node and not last_trans_set:
                print('insert trans_out node')
                last_trans_set = True
                transpose_out_node = onnx.helper.make_node('Transpose'
                                    , name = 'Transpose_out'
                                    , inputs = [last_ln_node+':0']
                                    , outputs = [start_node.output[0]]
                                    )
                transpose_out_node.attribute.insert(0, onnx.helper.make_attribute('perm', [1,0,2]))
                start_node.output[0] = last_ln_node+':0'
                graph.node.insert(node_index+1, transpose_out_node)
                continue

            if not Is_multihead_attention_start(graph, start_node):
                continue

            x = start_node.output[0]
            Multihead_attention_op_index += 1
            print ('fuse multi-head attention op : %d', Multihead_attention_op_index)

            for node in get_node_list_from_input_name(graph, x) :
                if node.op_type != 'Add' :
                    removed_node.add(node.name)
                    queue.append(node)
                    nodes_to_remove.append(node)

            in_project_weight = 'in_project_weight' + str(Multihead_attention_op_index)
            in_project_bias = ''
            out_project_weight = ''
            out_project_bias = ''
            y = ''
            num_heads = None
            check_ok = True

            query_bias = None
            key_bias = None
            value_bias = None
            while queue :
                node = queue.popleft()
                Add_node = get_node_from_input_name_and_op_type(graph, node.output[0], 'Add')
                if Add_node is not None and get_node_from_input_name_and_op_type(graph, Add_node.output[0], 'LayerNormalization') is not None:
                    # the last node
                    y = node.output[0]
                    continue
                elif get_node_from_output_name_and_op_type(graph, node.input[0], 'LayerNormalization') is not None:
                    #sometime the MHA sub_graph will extend some node out(like reshape), when it happens, stop the bfs and connect the graph 
                    ln_node = get_node_from_output_name_and_op_type(graph, node.input[0], 'LayerNormalization')
                    if not node_eq(ln_node, start_node):
                        ln_node.output[0] = node.output[0]
                        continue
                elif node.op_type == 'Softmax' :
                    mask_add_node = get_node_from_output_name_and_op_type(graph, node.input[0], 'Add')
                    if mask_add_node is not None:
                        l1 = get_node_list_from_input_name(graph, mask_add_node.input[0])
                        l2 = get_node_list_from_input_name(graph, mask_add_node.input[1])
                        if len(l1) > 1 and len(l2) == 1:
                            key_padding_mask_4d = mask_add_node.input[0]
                        elif len(l1) == 1 and len(l2) > 1:
                            key_padding_mask_4d = mask_add_node.input[1]
                        else:
                            key_padding_mask_4d = mask_add_node.input[1]
                            warnings.warn('the key_mask_padding before softmax seems strange, check it!')
                    else:
                        raise(Exception('can not find key_mask_padding_4d, time to update the fusion code'))
                elif node.op_type == 'Concat' and len(node.input) == 4:
                    num_heads_const_name = node.input[2]
                    if mother_graph is None:
                        num_heads_numpy = get_initializer_numpy_value(graph, num_heads_const_name)
                    else:
                        num_heads_numpy = get_initializer_numpy_value(mother_graph, num_heads_const_name)
                    num_heads = int(num_heads_numpy)

                elif node.op_type == 'MatMul' and recognise_node(node.name) == 1 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('query_weight is not init')
                        break

                    query_weight = get_initializer_numpy_value(graph, node.input[1])
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 2 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('key_weight is not init')
                        break
                    key_weight = get_initializer_numpy_value(graph, node.input[1])

                elif node.op_type == 'MatMul' and recognise_node(node.name) == 3 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('value_weight is not init')
                        break

                    value_weight = get_initializer_numpy_value(graph, node.input[1])
                
                elif node.op_type == 'Add' and recognise_node(node.name) == 1 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('query_bias is not init')
                        break

                    query_bias = get_initializer_numpy_value(graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 2 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('key_bias is not init')
                        break

                    key_bias = get_initializer_numpy_value(graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 3 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('value_bias is not init')
                        break

                    value_bias = get_initializer_numpy_value(graph, node.input[1])

                elif node.op_type == 'MatMul' and recognise_node(node.name) == 4 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('out_project_weight is not init')
                        break
 
                    out_project_weight = node.input[1]
                elif node.op_type == 'Add' and recognise_node(node.name) == 4 :
                    if not is_initializer(mother_graph, node.input[1]):
                        check_ok = False
                        print ('out_project_bias is not init')
                        break

                    out_project_bias = node.input[1]

                
                for next_node in get_node_list_from_input_name(graph, node.output[0]) :
                    if next_node.name not in removed_node :
                        queue.append(next_node)
                        removed_node.add(next_node.name)
                        nodes_to_remove.append(next_node)

            
            if check_ok :
                modify = True
                in_project_weight_cat = np.concatenate((query_weight, key_weight, value_weight), axis=1)
                in_project_weight_init = onnx.helper.make_tensor('in_project_weight'+str(Multihead_attention_op_index), TensorProto.FLOAT, list(in_project_weight_cat.shape), in_project_weight_cat.reshape(-1).tolist())
                if mother_graph is None:
                    append_initializer(graph, in_project_weight_init)
                else:
                    append_initializer(mother_graph, in_project_weight_init)
                if query_bias is None or key_bias is None or value_bias is None :
                    pass
                else :
                    in_project_bias = 'in_project_bias' + str(Multihead_attention_op_index)
                    in_project_bias_cat = np.concatenate((query_bias, key_bias, value_bias), axis=0)
                    in_project_bias_init = onnx.helper.make_tensor('in_project_bias'+str(Multihead_attention_op_index), TensorProto.FLOAT , list(in_project_bias_cat.shape), in_project_bias_cat.reshape(-1).tolist())
                    if mother_graph is None:
                        append_initializer(graph, in_project_bias_init)
                    else:
                        append_initializer(mother_graph, in_project_bias_init)

                Multihead_attention_node = onnx.helper.make_node('MultiHeadAttention'
                                                    , name = 'MultiHeadAttention_' + str(Multihead_attention_op_index)
                                                    , inputs = [x, key_padding_mask_2d, in_project_weight, in_project_bias, out_project_weight, out_project_bias, key_padding_mask_4d]
                                                    , outputs = [y]
                                                    )
                Multihead_attention_node.attribute.insert(0, onnx.helper.make_attribute("num_heads", num_heads, "num of heads"))
                start_index = get_node_index(graph, start_node)
                print('set Multihead_attention_node on start_node next pos: ', start_index)
                graph.node.insert(start_index+1, Multihead_attention_node)
                remove_node_list(graph, nodes_to_remove)
                break


        if not modify :
            break
    #bfs_remove(graph, 'gec_ged_model_revised_gedloss_1/body/parallel_0/body/embedding_to_padding/ToFloat', 'gec_ged_model_revised_gedloss_1/body/parallel_0/body/attention_bias_ignore_padding/ExpandDims_1')

    

def is_layernorm_start(graph, node):
    if get_node_from_output_name_and_op_type(graph, node.input[0], 'Add') is not None:
        return True
    return False


def is_layernorm_end(graph, node):
    next_list = get_node_list_from_input_name(graph, node.output[0])
    ln_node_list = ['Sub', 'Add', 'Sqrt', 'ReduceMean', 'Reciprocal', 'Mul']
    for next_node in next_list:
        if next_node.op_type not in ln_node_list:
            return True
    return False



def layer_normal_fusion_general(graph, mother_graph = None) :
    layer_normal_op_index = 0
    visited_node = set()
    if mother_graph is None:
        mother_graph = graph
    while True :
        #每次找到一个LN节点进行替换
        modify = False
        queue = collections.deque()
        for node_index in range(len(graph.node)):#get_node_topology(graph) :
            nodes_to_remove = []
            reduce_mean_node = graph.node[node_index]
            if reduce_mean_node.op_type != 'ReduceMean'  : #从ReduceMean节点出发
                continue
            if not is_layernorm_start(graph, reduce_mean_node) : ##输入为ReduceMean节点输出的node数量（ReduceMean的下个节点个数）
                continue
   
            x = reduce_mean_node.input[0] #原始保存输入
            visited_node.add(reduce_mean_node.name)
            queue.append(reduce_mean_node)
            nodes_to_remove.append(reduce_mean_node)
            alpha = ''
            bias = ''
            eps = 0.00001
            y = ''
            check_is_ln = True
            

            while queue :
                node = queue.popleft()

                if is_layernorm_end(graph, node):
                    # the last node
                    if node.op_type != 'Add':
                        raise Exception('last layernorm node is not Add')
                    if is_initializer(mother_graph, node.input[1]) :
                        bias = node.input[1]
                    y = node.output[0]
                    continue
                elif node.op_type == 'Mul' and is_initializer(mother_graph, node.input[1]):
                    alpha = node.input[1]
                elif (node.op_type == 'Sub' or node.op_type == 'Add') and get_node_from_input_name_and_op_type(graph, node.output[0], 'Sqrt') is None:
                    if is_initializer(mother_graph, node.input[1]):
                        bias = node.input[1]
                    elif is_initializer(mother_graph, node.input[0]):
                        bias = node.input[0]
                elif node.op_type == 'Add' and get_node_from_input_name_and_op_type(graph, node.output[0], 'Sqrt') is not None:
                    eps_numpy = get_initializer_numpy_value(mother_graph, node.input[1])
                    if eps_numpy is None :
                        print ('eps numpy is None')
                        check_is_ln = False
                        break
                    if eps_numpy.size != 1 :
                        print('eps numpy size != 1')
                        check_is_ln= False
                        break
                    eps = float(eps_numpy)
                for next_node in get_node_list_from_input_name(graph, node.output[0]) :
                    if next_node.name not in visited_node :
                        queue.append(next_node)
                        visited_node.add(next_node.name)
                        nodes_to_remove.append(next_node)
            if not check_is_ln :
                print ('check is ln False')
                continue
            layer_normal_op_index += 1
            print('find reduce mean node(%d) is LayerNormalization(%d)' % (node_index, layer_normal_op_index))
            modify = True
            layer_norm_node = onnx.helper.make_node('LayerNormalization'
                                                    , name = 'LayerNormalization_' + str(layer_normal_op_index)
                                                    , inputs = [x, alpha, bias]
                                                    , outputs = [y]
                                                    )
            layer_norm_node.attribute.insert(0, onnx.helper.make_attribute("axis", -1, "axis"))
            layer_norm_node.attribute.insert(1, onnx.helper.make_attribute("epsilon", eps, "eps for layer normal"))
            graph.node.insert(node_index, layer_norm_node)
            remove_node_list(graph, nodes_to_remove)
            break
        if not modify : #遍历整个计算图的循环如果不是因为修改LN跳出，那就是因为遍历结束了，完成任务跳出死循环
            break
    return graph

        
    
    

def subgraph_optimize(graph) :
    bfs_remove(graph, 'gec_ged_model_revised_gedloss_1/body/parallel_0/body/encoder/layer_3/ffn/concat', 'gec_ged_model_revised_gedloss_1/body/parallel_0/body/encoder/layer_3/ffn/Reshape__1193')
    for node in graph.node:
        if node.name == 'generic_loop_Loop__81':
            print('find decoder subgraph')
            sub_graph = node.attribute[0].g
            layer_normal_fusion_general(sub_graph, graph)
            Multihead_attention_fusion(sub_graph)
            
                

def Tranformer_fusion(graph):
    layer_normal_fusion_general(graph)
    Multihead_attention_fusion(graph)
    Tranformer_GELU_fusion(graph)
    FF_optimize(graph)

    
import onnxruntime as rt
import sys

def main() :
    print (sys.argv)
    if len(sys.argv) < 2 :
        print('Usage : input_dfsmnv2_onnx_file [out_stream_dfsmnv2_onnx_file]')
        return
    input_file = sys.argv[1]
    out_file = ''
    if len(sys.argv) > 2 :
        out_file = sys.argv[2]
    else :
        out_file = input_file.split('.onnx')[0] + '-stream.onnx'
    #onnx run time optimize
    #onnx_tmp_file = input_file.split('.onnx')[0] + '-runtimeopt.onnx'

    # sess_options = rt.SessionOptions()
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC #
    # sess_options.optimized_model_filepath = onnx_tmp_file
    # session = rt.InferenceSession(input_file, sess_options)

    model = onnx.load(input_file)
    #layer_normal_fusion_general(model.graph)
    Tranformer_fusion(model.graph)

    
    #subgraph_optimize(model.graph)

    
    
    # Reshape_fusion(model.graph)
    


    # PantherFsmnV2_fusion(model.graph)
    # transpose_fusion(model.graph)
    # PantherDfsmnv2_fusion(model.graph)
    # transpose_fusion(model.graph)
    # add_fsmn_state(model.graph)


    onnx.save(model, out_file)
    print('success convert, save out file to %s' % out_file)

if __name__ == '__main__' :
    main()
