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
import collections
import pdb
import warnings
import sys

def get_node_topology(graph, mother_graph=None) : 
    #返回所有可计算节点 index 的拓扑排序
    node_num = len(graph.node)
    compute_tensor_name_list = set()#input名称集合
    if mother_graph is not None:
        return range(len(graph.node))
    for input in graph.input :
        compute_tensor_name_list.add(input.name)
    topology_list = []
    print('total node num : %d' % node_num)
    initializer_name_set = set() #权重名称集合
    for init in graph.initializer :
        initializer_name_set.add(init.name)
    while len(topology_list) < node_num :
        print(len(topology_list))
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
                # if len(topology_list) == 134:
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
def is_initializer(graph, name, mother_graph = None) :
    if get_initializer(graph, name) != None :
        return True
    if mother_graph is None :
        return False
    else :
        return get_initializer(mother_graph, name) != None


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
def get_initializer_constant_numpy_value(graph, name) :
    init_tensor = get_initializer(graph, name)
    if init_tensor:
        return TensorProtoToNumpy(init_tensor)
    for node in graph.node:
        if node.op_type == "Constant":
            if node.output[0] == name:
                attr_numpy = TensorProtoToNumpy(node.attribute[0].t)
                graph.node.remove(node)
                return attr_numpy
    return None

   
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
def is_layernorm_start(graph, node):
    if get_node_from_input_name_and_op_type(graph, node.output[0], 'Sub'):
        return True
    return False
def is_layernorm_end(graph, node):
    pre_list = get_node_list_from_output_name(graph, node.input[0])
    if node.op_type == "Add" and len(pre_list) == 1 and pre_list[0].op_type == "Mul":
        return True
    return False

def Tranformer_GELU_fusion(graph, mother_graph = None) :
    gelu_index = 0
    while True :
        modify = False
        for node_index in get_node_topology(graph, mother_graph=mother_graph) :
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
    if gelu_index > 0 :
        return True
    else :
        return False



def bfs_remove(graph, start_node, end_node, connect = False, verbose = False) :
    """
    remove nodes from start_node to end_node with bfs, removed nodes include those two nodes

    start_node: name of the start_node
    end_node: name of the end_node, when it's '', scan to the end
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
        if node.name == end_node and end_node != '':
            continue
        i = 0
        if node.op_type == 'Loop':
            i = 1
        for next_node in get_node_list_from_input_name(graph, node.output[i]) :
            if next_node.name not in removed_node :
                queue.append(next_node)
                removed_node.add(next_node.name)
                nodes_to_remove.append(next_node)
    if verbose:
        print('bfs remove total', len(nodes_to_remove), 'nodes')
    remove_node_list(graph, nodes_to_remove)



def FF_optimize(graph, mother_graph = None) :
    #优化feed forward部分
    print('*********FeedForward optimizing*********')
    visited_node = set()
    not_remove_op = ['MatMul', 'Add', 'Gelu', 'Relu', 'Tanh']
    last_ln_node_name = get_last_ln_node_name(graph)
    while True :
        modify = False
        queue = collections.deque()
        for node_index in get_node_topology(graph, mother_graph=mother_graph) :
            nodes_to_remove = []
            ln_node = graph.node[node_index]
            if ln_node.op_type != 'LayerNormalization' or ln_node.name == last_ln_node_name or get_node_from_input_name_and_op_type(graph, ln_node.output[0], 'MultiHeadAttention') is not None: 
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
                        continue
                    bias_node = get_node_from_input_name_and_op_type(graph, reshape_node.output[0], 'Add')
                    if bias_node is None:
                        continue
                    bias_node.input[0] = node.output[0]

                    activate_node = get_node_from_input_name_and_op_type(graph, bias_node.output[0], 'Gelu')
                    if activate_node is None:
                        activate_node = get_node_from_input_name_and_op_type(graph, bias_node.output[0], 'Relu')
                    if activate_node is None:
                        last_matmul = node
                        last_bias = bias_node
                    else:
                        node.input[0] = x
                        activate_output = activate_node.output[0]

                next_list = get_node_list_from_input_name(graph, node.output[0])
                if node.op_type == 'Loop':
                    next_list = get_node_list_from_input_name(graph, node.output[1])

                for next_node in next_list:
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


def reverse_bfs(graph, start_outname, end_name = [], end_type = []) :
    visited_node = set()
    queue = collections.deque()
    start_list = get_node_list_from_output_name(graph, start_outname)
    for node in start_list :
        queue.append(node)
        visited_node.add(node.name)
    while queue :
        node = queue.popleft()
        if node.op_type in end_type or node.name in end_name:
            return node
        next_list = []
        for i in range(len(node.input)):
            next_list.extend(get_node_list_from_output_name(graph, node.input[i]))
        for next_node in next_list :
            if next_node.name not in visited_node :
                queue.append(next_node)
                visited_node.add(next_node.name)
    return False

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

def get_last_ln_node_name(graph) :
    last_ln_node_name = ''
    max_ln_id = -1
    for node in graph.node:
        if node.op_type == 'LayerNormalization':
            if int(node.name.split('_')[-1]) > max_ln_id:
                max_ln_id = int(node.name.split('_')[-1])
                last_ln_node_name = node.name
    return last_ln_node_name


def Multihead_attention_fusion(graph) :
    Multihead_attention_op_index = 0
    removed_node = set()
    #step1: make Multihead_attention's mask
    #key_padding_mask_bool: Multihead_attention needs the mask input like: [f,f,f,f,f,t,t,t], where t(True) is the padding token
    #key_padding_mask_float: this mask directly add to the input of softmax
    #key_padding_mask: the combination of two mask above
    key_padding_mask =  ''
    for node in graph.node:
        if node.op_type == 'Softmax' :
            mask_add_node = get_node_from_output_name_and_op_type(graph, node.input[0], 'Add')
            if mask_add_node is not None:
                l1 = get_node_list_from_input_name(graph, mask_add_node.input[0])
                l2 = get_node_list_from_input_name(graph, mask_add_node.input[1])
                if len(l1) > 1 and len(l2) == 1:
                    key_padding_mask = mask_add_node.input[0]
                elif len(l1) == 1 and len(l2) > 1:
                    key_padding_mask = mask_add_node.input[1]
                else:
                    key_padding_mask = mask_add_node.input[1]
                    warnings.warn('the key_mask_padding before softmax seems strange, check it!')
            else:
                warnings.warn('this transformer has no attn mask')
            print('deal key_padding_mask ok')
            break

    #step2: find the last LN node in advance so that can insert output transpose node
    last_ln_node_name = get_last_ln_node_name(graph)
    print('last ln node:', last_ln_node_name)


    #step3: BFS the MHA graph and get all the initialiser
    #the input usually is B,T,E, but the MHA needs T,B,E so we need two transpose nodes(in & out)
    first_trans_set = False
    last_trans_set = False
    while True :
        modify = False
        queue = collections.deque()
        for node_index in get_node_topology(graph) :
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
            if start_node.name == last_ln_node_name and not last_trans_set:
                print('insert trans_out node')
                last_trans_set = True
                transpose_out_node = onnx.helper.make_node('Transpose'
                                    , name = 'Transpose_out'
                                    , inputs = [last_ln_node_name+':0']
                                    , outputs = [start_node.output[0]]
                                    )
                transpose_out_node.attribute.insert(0, onnx.helper.make_attribute('perm', [1,0,2]))
                start_node.output[0] = last_ln_node_name+':0'
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
            found_num_head = False
            while queue :
                node = queue.popleft()
                is_last_node = False
                Add_node = get_node_from_input_name_and_op_type(graph, node.output[0], 'Add')
                if Add_node is not None and get_node_from_input_name_and_op_type(graph, Add_node.output[0], 'LayerNormalization') is not None:
                    # the last node
                    y = node.output[0]
                    is_last_node = True
                if get_node_from_output_name_and_op_type(graph, node.input[0], 'LayerNormalization') is not None:
                    #sometime the MHA sub_graph will extend some node out(like reshape), when it happens, stop the bfs and connect the graph 
                    ln_node = get_node_from_output_name_and_op_type(graph, node.input[0], 'LayerNormalization')
                    if not node_eq(ln_node, start_node):
                        ln_node.output[0] = node.output[0]
                        continue
                elif node.op_type == 'Concat' and len(node.input) == 4 and not found_num_head:
                    num_heads_const_name = node.input[2]
                    if is_initializer(graph, num_heads_const_name) :
                        target_graph = graph
                    else:
                        raise(Exception("num heads is None"))
                    num_heads_numpy = get_initializer_constant_numpy_value(target_graph, num_heads_const_name)
                    num_heads = int(num_heads_numpy)
                    found_num_head = True

                elif node.op_type == 'MatMul' and recognise_node(node.name) == 1 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('query_weight is not init')
                        break

                    query_weight = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 2 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('key_weight is not init')
                        break
                    key_weight = get_initializer_constant_numpy_value(target_graph, node.input[1])

                elif node.op_type == 'MatMul' and recognise_node(node.name) == 3 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('value_weight is not init')
                        break
                    value_weight = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 1 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('query_bias is not init')
                        break
                    query_bias = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 2 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('key_bias is not init')
                        break
                    key_bias = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 3 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('value_bias is not init')
                        break
                    value_bias = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 4 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('out_project_weight is not init')
                        break
                    out_project_weight = node.input[1]
                elif node.op_type == 'Add' and recognise_node(node.name) == 4 :
                    if is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('out_project_bias is not init')
                        break
                    out_project_bias = node.input[1]
                if is_last_node == True:
                  continue
                for next_node in get_node_list_from_input_name(graph, node.output[0]) :
                    if next_node.name not in removed_node :
                        queue.append(next_node)
                        removed_node.add(next_node.name)
                        nodes_to_remove.append(next_node)
            if check_ok :
                modify = True
                in_project_weight_cat = np.concatenate((query_weight, key_weight, value_weight), axis=1)
                in_project_weight_init = onnx.helper.make_tensor('in_project_weight'+str(Multihead_attention_op_index), TensorProto.FLOAT, list(in_project_weight_cat.shape), in_project_weight_cat.reshape(-1).tolist())
                append_initializer(graph, in_project_weight_init)
                if query_bias is None or key_bias is None or value_bias is None :
                    pass
                else :
                    in_project_bias = 'in_project_bias' + str(Multihead_attention_op_index)
                    in_project_bias_cat = np.concatenate((query_bias, key_bias, value_bias), axis=0)
                    in_project_bias_init = onnx.helper.make_tensor('in_project_bias'+str(Multihead_attention_op_index), TensorProto.FLOAT , list(in_project_bias_cat.shape), in_project_bias_cat.reshape(-1).tolist())
                    append_initializer(graph, in_project_bias_init)

                Multihead_attention_node = onnx.helper.make_node('MultiHeadAttention'
                                                    , name = 'MultiHeadAttention_' + str(Multihead_attention_op_index)
                                                    , inputs = [x, key_padding_mask, in_project_weight, in_project_bias, out_project_weight, out_project_bias]
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







def layer_normal_fusion_general(graph, mother_graph = None) :

    for node in graph.node:
        if node.op_type == "Identity":
            if len(node.output) == 1 and len(node.input) == 1 and is_initializer(graph, node.input[0]):
                next_list = get_node_list_from_input_name(graph, node.output[0])
                for n_node in next_list:
                    for i in range(len(n_node.input)):
                        if n_node.input[i] == node.output[0]:
                            n_node.input[i] = node.input[0]
    layer_normal_op_index = 0
    visited_node = set()
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
            axes = get_attribute_ints_value(reduce_mean_node, "axes")[0]
            visited_node.add(reduce_mean_node.name)
            queue.append(reduce_mean_node)
            nodes_to_remove.append(reduce_mean_node)
            alpha = ''
            bias = ''
            eps = 0.00001
            y = ''
            check_is_ln = True

            x = reduce_mean_node.input[0] #原始保存输入
            while queue :
                node = queue.popleft()
                if is_layernorm_end(graph, node):
                    if is_initializer(graph, node.input[1], mother_graph) :
                        bias = node.input[1]
                    else :
                        # identity_node = get_node_from_output_name_and_op_type(graph, node.input[1], "Identity")
                        # if identity_node and is_initializer(graph, identity_node.input[0], mother_graph):
                        #     bias = identity_node.input[0]
                        # else :
                        raise(Exception("last node has no bias"))
                    y = node.output[0]
                    continue
                elif node.op_type == 'Mul' and is_initializer(graph, node.input[1], mother_graph):
                    alpha = node.input[1]
                elif node.op_type == 'Add' and get_node_from_input_name_and_op_type(graph, node.output[0], 'Sqrt'):
                    if mother_graph is None :
                        eps_numpy = get_initializer_constant_numpy_value(graph, node.input[1])
                    elif get_initializer_constant_numpy_value(graph, node.input[1]) is None:
                        eps_numpy = get_initializer_constant_numpy_value(mother_graph, node.input[1])
                    else :
                        eps_numpy = get_initializer_constant_numpy_value(graph, node.input[1])
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
            print('find reduce mean node(%d) is LayerNorm(%d)' % (node_index, layer_normal_op_index))
            modify = True
            layer_norm_node = onnx.helper.make_node('LayerNorm'
                                                    , name = 'LayerNorm_' + str(layer_normal_op_index)
                                                    , inputs = [x, alpha, bias]
                                                    , outputs = [y]
                                                    )
            layer_norm_node.domain = "pmx"
            layer_norm_node.attribute.insert(0, onnx.helper.make_attribute("axes", axes, "axes"))
            #layer_norm_node.attribute.insert(1, onnx.helper.make_attribute("elementwise_affine", 1, "elementwise_affine"))
            layer_norm_node.attribute.insert(1, onnx.helper.make_attribute("eps", eps, "eps for layer normal"))
            graph.node.insert(node_index, layer_norm_node)
            remove_node_list(graph, nodes_to_remove)
            break
        if not modify : #遍历整个计算图的循环如果不是因为修改LN跳出，那就是因为遍历结束了，完成任务跳出死循环
            break
    return graph

# def remove_Transpose2(graph, mother_graph = None):
#     print('fixing transpse2')
#     cnt = 0
#     for node in graph.node:
#         if node.op_type == 'Transpose2':
#             concat_node = get_node_from_output_name_and_op_type(graph, node.input[1], 'Concat')
#             if concat_node is not None and is_initializer(graph, concat_node.input[0], mother_graph) and is_initializer(graph, concat_node.input[1], mother_graph):
#                 transpose2_list = get_node_list_from_input_name_and_op_type(graph, concat_node.output[0], 'Transpose2')
#                 for n in transpose2_list:
#                     cnt += 1
#                     n.input.remove(n.input[1])
#                     bfs_remove(graph, n.name, n.name, connect=True, verbose=False)
#                 remove_node(graph, concat_node)
#                 print('concat_node has been removed')
#     print(cnt," Transpose2 nodes have been fixed")

# def subgraph_optimize(graph) :
#     for node in graph.node:
#         if node.op_type == 'Loop':
#             print('find decoder subgraph')
#             sub_graph = node.attribute[0].g
#             #remove_Transpose2(sub_graph, mother_graph = graph)
#             layer_normal_fusion_general(sub_graph, mother_graph = graph)
#             DecoderTransformerFusion(sub_graph, mother_graph=graph)
#             # Multihead_attention_fusion(sub_graph, mother_graph = graph)
#             # Tranformer_GELU_fusion(sub_graph, sub_graph=True)
#             # FF_optimize(sub_graph, mother_graph=graph)

# def subgraph_dump(graph, first = True) :
#     for node in graph.node:
#         if node.op_type == 'Loop':
#             sub_graph = node.attribute[0].g
#             for node in sub_graph.node :
#                 if node.op_type == 'If':
#                     sub_graph_else = node.attribute[0].g
#                     sub_graph_then = node.attribute[1].g
#                     model_else = onnx.helper.make_model(sub_graph_else)
#                     model_then = onnx.helper.make_model(sub_graph_then)
#                     model_name_else = '/Users/yangyuehang/Documents/model/correct/' + 'else' + '.onnx'
#                     model_name_then = '/Users/yangyuehang/Documents/model/correct/' + 'then' + '.onnx'
#                     onnx.save(model_else, model_name_else)
#                     onnx.save(model_then, model_name_then)
#             model = onnx.helper.make_model(sub_graph)
#             if first :
#                 model_name = '/Users/yangyuehang/Documents/model/correct/' + 'original_loop' + '.onnx'
#             else :
#                 model_name = '/Users/yangyuehang/Documents/model/correct/' + node.name.split('/')[-1]+ '_decoder' + '.onnx'
#             onnx.save(model, model_name)


def Tranformer_fusion(graph):
    layer_normal_fusion_general(graph)
    Multihead_attention_fusion(graph)
    Tranformer_GELU_fusion(graph)
    FF_optimize(graph)



def JudgeLastNode(graph, node) :
    if node.op_type != 'Add' or get_node_from_input_name_and_op_type(graph, node.output[0], 'LayerNormalization') is None :
        return False
    if get_node_from_input_name_and_op_type(graph, node.input[1], 'LayerNormalization') :
        start_outname = node.input[0]
    elif get_node_from_input_name_and_op_type(graph, node.input[0], 'LayerNormalization') :
        start_outname = node.input[1]
    else :
        raise Exception('something wrong with skip add node')
    if reverse_bfs(graph, start_outname, end_type=['Relu', 'Gelu']) :
        return True
    return False

    





def DecoderTransformerFusion(graph, mother_graph = None) :
    fuse_subgraph = True
    if mother_graph is None:
        mother_graph = graph
        fuse_subgraph = False
    decoder_transformer_op_index = 0
    removed_node = set()
    gelu_flg = Tranformer_GELU_fusion(graph, mother_graph=mother_graph)

    # DecoderTransformer inputs
    x = ''
    key_padding_mask1 = '' # first(mask) MHA's mask
    key_padding_mask2 = '' # second MHA's mask
    history_K = ''
    history_V = ''
    encoder_K = ''
    encoder_V = ''
    # layer normalization's params
    input_normal_wei = ''
    input_normal_bias = ''
    first_normal_wei = ''
    first_normal_bias = ''
    second_normal_wei = ''
    second_normal_bias = ''
    # first MHA's params, include Q,K,V
    in_project_weight1 = '' 
    in_project_bias1 = ''
    # second MHA's params, only Q
    in_project_weight2 = ''
    in_project_bias2 = ''
    # output linear trans' params
    out_project_weight1 = ''
    out_project_weight2 = ''
    out_project_bias1 = ''
    out_project_bias2 = ''
    
    # feed forward params
    ff_weight1 = ''
    ff_weight2 = ''
    ff_bias1 = ''
    ff_bias2 = ''
    num_heads = None

    # DecoderTransformer outputs
    y = ''
    history_K_out = ''
    history_V_out = ''


    #step1: make Multihead_attention's mask
    #key_padding_mask_bool: Multihead_attention needs the mask input like: [f,f,f,f,f,t,t,t], where t(True) is the padding token
    #key_padding_mask_float: this mask directly add to the input of softmax
    #key_padding_mask: the combination of two mask above
    
    for node in graph.node:
        if node.op_type == 'Softmax' :
            mask_add_node = get_node_from_output_name_and_op_type(graph, node.input[0], 'Add')
            if mask_add_node is not None:
                l1 = get_node_list_from_input_name(graph, mask_add_node.input[0])
                l2 = get_node_list_from_input_name(graph, mask_add_node.input[1])
                if len(l1) > 1 and len(l2) == 1:
                    if get_node_from_output_name_and_op_type(graph, mask_add_node.input[0], 'Slice') is not None :
                        key_padding_mask1 = mask_add_node.input[0]
                    else :
                        key_padding_mask2 = mask_add_node.input[0]
                elif len(l1) == 1 and len(l2) > 1:
                    if get_node_from_output_name_and_op_type(graph, mask_add_node.input[1], 'Slice') is not None :
                        key_padding_mask1 = mask_add_node.input[1]
                    else :
                        key_padding_mask2 = mask_add_node.input[1]
                else:
                    if get_node_from_output_name_and_op_type(graph, mask_add_node.input[1], 'Slice') is not None :
                        key_padding_mask1 = mask_add_node.input[1]
                    else :
                        key_padding_mask2 = mask_add_node.input[1]
                    warnings.warn('the key_mask_padding before softmax seems strange, check it!')
            else:
                raise(Exception('can not find key_mask_padding_bool, time to update the fusion code'))
            if key_padding_mask1 != '' and key_padding_mask2 != '' :
                print('have found mask 1&2')
                break

    #step2: find the last LN node in advance so that can insert output transpose node
    last_ln_node_name = get_last_ln_node_name(graph)
    print('last ln node:', last_ln_node_name)


    #step3: BFS the graph and get all the initialiser
    #the input usually is B,T,E, but the Transformer needs T,B,E so we need two transpose nodes(in & out)
    first_trans_set = False
    last_trans_set = False
    while True :
        modify = False
        queue = collections.deque()
        found_activation = False
        for node_index in get_node_topology(graph, mother_graph) :
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

            if start_node.name == last_ln_node_name and not last_trans_set:
                print('insert trans_out node')
                last_trans_set = True
                transpose_out_node = onnx.helper.make_node('Transpose'
                                    , name = 'Transpose_out'
                                    , inputs = [last_ln_node_name+':0']
                                    , outputs = [start_node.output[0]]
                                    )
                transpose_out_node.attribute.insert(0, onnx.helper.make_attribute('perm', [1,0,2]))
                start_node.output[0] = last_ln_node_name+':0'
                graph.node.insert(node_index+1, transpose_out_node)
                continue

            if not Is_multihead_attention_start(graph, start_node):
                continue
            x = start_node.input[0]
            decoder_transformer_op_index += 1
            print ('fuse decoder_transformer op : %d', decoder_transformer_op_index)

            for node in get_node_list_from_input_name(graph, x) :
                if node.op_type != 'Add' :
                    removed_node.add(node.name)
                    queue.append(node)
                    nodes_to_remove.append(node)

            check_ok = True
            query_bias = None
            key_bias = None
            value_bias = None
            found_num_head = False
            while queue :
                node = queue.popleft()
                is_last_node = False
                if JudgeLastNode(graph, node):
                    # the last node
                    y = node.output[0]
                    is_last_node = True
                if get_node_from_output_name_and_op_type(graph, node.input[0], 'LayerNormalization') is not None:
                    #sometime the MHA sub_graph will extend some node out(like reshape), when it happens, stop the bfs and connect the graph 
                    ln_node = get_node_from_output_name_and_op_type(graph, node.input[0], 'LayerNormalization')
                    if ln_node.name == last_ln_node_name:
                        ln_node.output[0] = node.output[0]
                        continue
                elif node.op_type == 'LayerNormalization' :
                    if int(node.name.split('_')[-1]) % 3 == 1 :
                        input_normal_wei = node.input[1]
                        input_normal_bias = node.input[2]
                    elif int(node.name.split('_')[-1]) % 3 == 2 :
                        first_normal_wei = node.input[1]
                        first_normal_bias = node.input[2]
                    elif int(node.name.split('_')[-1]) % 3 == 0 :
                        second_normal_wei = node.input[1]
                        second_normal_bias = node.input[2]
                elif node.op_type == 'Relu' or node.op_type == 'Gelu':
                    found_activation = True
                elif node.op_type == 'Concat' and len(node.input) == 4 and not found_num_head:
                    # find num_heads
                    num_heads_const_name = node.input[2]
                    if is_initializer(mother_graph, num_heads_const_name):
                        target_graph = mother_graph
                    elif is_initializer(graph, num_heads_const_name) :
                        target_graph = graph
                    else:
                        num_heads_const_name = node.input[1]
                        if is_initializer(mother_graph, num_heads_const_name):
                            target_graph = mother_graph
                        elif is_initializer(graph, num_heads_const_name) :
                            target_graph = graph
                        else:
                            raise(Exception("num heads is None"))
                    num_heads_numpy = get_initializer_constant_numpy_value(target_graph, num_heads_const_name)
                    num_heads = int(num_heads_numpy)
                    print('find num heads: %d' % num_heads)
                    found_num_head = True
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 1 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('query_weight is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    if int(ln_above.name.split('_')[-1]) % 3 == 1 :
                        query_weight = get_initializer_constant_numpy_value(target_graph, node.input[1])
                    elif int(ln_above.name.split('_')[-1]) % 3 == 2 :
                        in_project_weight2 = node.input[1]
                    else :
                        raise Exception('something with ln_above number')
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 2 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('key_weight is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    assert(int(ln_above.name.split('_')[-1]) % 3 == 1)
                    key_weight = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 3 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('value_weight is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    assert(int(ln_above.name.split('_')[-1]) % 3 == 1)
                    value_weight = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 1 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('query_bias is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    if int(ln_above.name.split('_')[-1]) % 3 == 1 :
                        query_bias = get_initializer_constant_numpy_value(target_graph, node.input[1])
                    elif int(ln_above.name.split('_')[-1]) % 3 == 2 :
                        in_project_bias2 = node.input[1]
                    else :
                        raise Exception('something with ln_above number')
                elif node.op_type == 'Add' and recognise_node(node.name) == 2 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('key_bias is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    assert(int(ln_above.name.split('_')[-1]) % 3 == 1)
                    key_bias = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'Add' and recognise_node(node.name) == 3 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('value_bias is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    assert(int(ln_above.name.split('_')[-1]) % 3 == 1)
                    value_bias = get_initializer_constant_numpy_value(target_graph, node.input[1])
                elif node.op_type == 'MatMul' and recognise_node(node.name) == 4 :
                    # find out_project_weight
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('out_project_weight is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    if int(ln_above.name.split('_')[-1]) % 3 == 1 :
                        out_project_weight1 = node.input[1]
                    elif int(ln_above.name.split('_')[-1]) % 3 == 2 :
                        out_project_weight2 = node.input[1]
                    else :
                        raise Exception('something with ln_above number')
                elif node.op_type == 'Add' and recognise_node(node.name) == 4 :
                    if is_initializer(mother_graph, node.input[1]):
                        target_graph = mother_graph
                    elif is_initializer(graph, node.input[1]) :
                        target_graph = graph
                    else:
                        check_ok = False
                        print ('out_project_bias is not init')
                        break
                    ln_above = reverse_bfs(graph, node.output[0], end_type=['LayerNormalization'])
                    assert(ln_above != False)
                    if int(ln_above.name.split('_')[-1]) % 3 == 1 :
                        out_project_bias1 = node.input[1]
                    elif int(ln_above.name.split('_')[-1]) % 3 == 2 :
                        out_project_bias2 = node.input[1]
                    else :
                        raise Exception('something with ln_above number')
                elif node.op_type == 'Concat' and len(node.input) == 2 and len(get_node_list_from_input_name(graph, node.output[0])) >= 3:
                    # find history K, V
                    is_last_node = True 
                    if get_node_from_input_name_and_op_type(graph, node.output[0], 'MatMul') is not None :
                        history_V = node.input[0]
                        history_V_out = node.output[0]
                    elif get_node_from_input_name_and_op_type(graph, node.output[0], 'Reshape') is not None :
                        history_K = node.input[0]
                        history_K_out = node.output[0]
                elif node.op_type == 'MatMul' and int(reverse_bfs(graph, node.output[0], end_type=['LayerNormalization']).name[-1]) % 3 == 2 :
                    # find encoder K, V
                    if get_node_from_output_name_and_op_type(graph, node.input[0], 'Softmax') is not None :
                        reshape_node = get_node_from_output_name_and_op_type(graph, node.input[1], 'Reshape')
                        if reshape_node is not None and reshape_node.input[0].find('Identity') != -1 :
                            encoder_V = reshape_node.output[0]
                    else :
                        reshape_node1 = get_node_from_output_name_and_op_type(graph, node.input[0], 'Reshape')
                        if reshape_node1 is not None :
                            reshape_node2 = get_node_from_output_name_and_op_type(graph, reshape_node1.input[0], 'Reshape')
                            if reshape_node2 is not None and reshape_node2.input[0].find('Identity') != -1 :
                                encoder_K = reshape_node2.output[0]
                elif node.op_type == 'MatMul' and int(reverse_bfs(graph, node.output[0], end_type=['LayerNormalization']).name[-1]) % 3 == 0 :
                    if not found_activation :
                        ff_weight1 = node.input[1]
                    else :
                        ff_weight2 = node.input[1]
                elif node.op_type == 'Add' and int(reverse_bfs(graph, node.output[0], end_type=['LayerNormalization']).name[-1]) % 3 == 0 :
                    if not found_activation :
                        ff_bias1 = node.input[1]
                    else :
                        ff_bias2 = node.input[1]

                if is_last_node == True:
                    continue
                for next_node in get_node_list_from_input_name(graph, node.output[0]) :
                    if next_node.name not in removed_node :
                        queue.append(next_node)
                        removed_node.add(next_node.name)
                        nodes_to_remove.append(next_node)
            if check_ok :
                modify = True
                in_project_weight1 = 'in_project_weight'+str(decoder_transformer_op_index)
                in_project_weight_cat = np.concatenate((query_weight, key_weight, value_weight), axis=1)
                in_project_weight_init = onnx.helper.make_tensor(in_project_weight1, TensorProto.FLOAT, list(in_project_weight_cat.shape), in_project_weight_cat.reshape(-1).tolist())
                append_initializer(mother_graph, in_project_weight_init)
                if query_bias is None or key_bias is None or value_bias is None :
                    pass
                else :
                    in_project_bias1 = 'in_project_bias' + str(decoder_transformer_op_index)
                    in_project_bias_cat = np.concatenate((query_bias, key_bias, value_bias), axis=0)
                    in_project_bias_init = onnx.helper.make_tensor('in_project_bias'+str(decoder_transformer_op_index), TensorProto.FLOAT , list(in_project_bias_cat.shape), in_project_bias_cat.reshape(-1).tolist())
                    append_initializer(mother_graph, in_project_bias_init)

                DecoderTransformer_node = onnx.helper.make_node('DecoderTransformer'
                                                    , name = 'DecoderTransformer_' + str(decoder_transformer_op_index)
                                                    , inputs = [x, key_padding_mask1, key_padding_mask2, input_normal_wei, input_normal_bias, first_normal_wei, first_normal_bias, second_normal_wei, second_normal_bias,
                                                    history_K, history_V, encoder_K, encoder_V, in_project_weight1, in_project_bias1, in_project_weight2, in_project_bias2, out_project_weight1, out_project_bias1, out_project_weight2, out_project_bias2, ff_weight1, ff_bias1, ff_weight2, ff_bias2]
                                                    , outputs = [y, history_K_out, history_V_out]
                                                    )
                DecoderTransformer_node.attribute.insert(0, onnx.helper.make_attribute("num_heads", num_heads, "num of heads"))
                DecoderTransformer_node.attribute.insert(0, onnx.helper.make_attribute("gelu_flg", gelu_flg, "gelu or relu"))
                start_index = get_node_index(graph, start_node)
                print('set DecoderTransformer_node on start_node next pos: ', start_index)
                graph.node.insert(start_index+1, DecoderTransformer_node)
                remove_node_list(graph, nodes_to_remove)
                break


        if not modify :
            break




def main() :

    input_file = 'C:\yangyuehang\桌面\onnx_model\\evoformer_1_dym.onnx'
    out_file = ''
    if len(sys.argv) > 2 :
        input_file = sys.argv[1]
        out_file = sys.argv[2]
    else :
        out_file = 'C:\yangyuehang\桌面\onnx_model\\evoformer_1_dym-new.onnx'


    add_pmx = True
    model = onnx.load(input_file)

    for ot in model.opset_import:
        if ot.domain == 'pmx':
            add_pmx = False
            break

    if add_pmx:
        model.opset_import.append(model.opset_import[0])
        model.opset_import[-1].domain = 'pmx'
        model.opset_import[-1].version = 1
    #fix_graph(model.graph)
    #fix_MatrixBandPart(model.graph)
    #fix_graph(model.graph)

    layer_normal_fusion_general(model.graph)
    get_node_topology(model.graph, None)


    #remove_Transpose2(model.graph)
    #bfs_remove(model.graph, 'gec_ged_model_revised_gedloss_1/body/parallel_0/body/encoder/attention_bias_to_padding/Less', '')
    # Tranformer_fusion(model.graph)
    # subgraph_optimize(model.graph)
    # subgraph_dump(model.graph, first=False)


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
