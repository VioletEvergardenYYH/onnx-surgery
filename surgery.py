import functools
import onnx
import numpy as np
from onnx import helper
from onnx import numpy_helper
import pdb
import collections
def get_node_topology(graph) : 
    #返回所有可计算节点 index 的拓扑排序
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
        # if node_num == 619:
        #     print(topology_list)

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
                # if node_num == 619:
                #     print(node.name)
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
def remove_node(graph, node) :
    graph.node.remove(node)
def remove_node_list(graph, node_list) :
    for node in node_list :
        remove_node(graph, node)
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
def export_graph(graph, name):
    model_def = helper.make_model(graph, producer_name='onnx-example')
    onnx.save(model_def, name)
def fix_IsFinite(graph) :
        op_index = 0
        while True :
            modify = False
            for node_index in range(len(graph.node)) :
                IsFinite_node = graph.node[node_index]
                if IsFinite_node.op_type != 'IsFinite' :
                    continue
            
                op_index += 1
                print('find IsFinite_node'+str(op_index))
                x = IsFinite_node.input[0]
                y = IsFinite_node.output[0]
                isnan_out = 'IsNaN'+str(op_index)+':0'
                isinf_out = 'IsInf'+str(op_index)+':0'

                isnan_node = onnx.helper.make_node('IsNaN'
                                                    ,name='IsNaN'+str(op_index)
                                                    ,inputs=[x]
                                                    ,outputs=[isnan_out]
                                                    )

                isinf_node = onnx.helper.make_node('IsInf'
                                                    , name = 'IsInf'+str(op_index)
                                                    , inputs = [x]
                                                    , outputs = [isinf_out]
                                                    )
 
                or_node = onnx.helper.make_node('Or'
                                                    , name = 'Or'+str(op_index)
                                                    , inputs = [isnan_out, isinf_out]
                                                    , outputs = ['Or'+str(op_index)+':0']
                                                    )
                not_node = onnx.helper.make_node('Not'
                                                    , name = 'Not'+str(op_index)
                                                    , inputs = ['Or'+str(op_index)+':0']
                                                    , outputs = [y]
                                                    )
                graph.node.insert(node_index, isnan_node)
                graph.node.insert(node_index, isinf_node)
                graph.node.insert(node_index, or_node)
                graph.node.insert(node_index, not_node)
                graph.node.remove(IsFinite_node)
                modify = True
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
                    if last_node.op_type == 'MatMul' and node.op_type == 'convert_gradient_to_tensor_HBc3xYw22Mw':
                        
                        for node_index in range(len(graph.node)):
                            mnode = graph.node[node_index]
                            if node_eq(last_node, mnode):
                                break
                        reshape_node = helper.make_node('Reshape'
                                            ,name='insert_Reshape'
                                            ,inputs=[x, 'const_fold_cg']
                                            ,outputs=['insert_Reshape:0']
                                            )
                        graph.node.insert(node_index, reshape_node)
                        shape_init = onnx.helper.make_tensor('const_fold_cg', 7, [2], [512, -1])
                        append_initializer(graph, shape_init)
                        last_node.input[1] = 'insert_Reshape:0'
                
                    else:
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
class Surgery(object):
    def __init__(self, onnx_model_path):
        self.model = onnx.load(onnx_model_path)

    def export(self, file_name, infer_shapes=False):
        if infer_shapes:
            self.model = onnx.shape_inference.infer_shapes(self.model)
        #onnx.checker.check_model(self.model)
        onnx.save(self.model, file_name)

    def list_model_inputs(self, nums):
        count = 0
        for mi in self.model.graph.input:
            print(mi)
            '''
            # NOTE:
            # the shape or dim in tensor is something like this below
            # it is just a list of {}, both dim_param and dim_value are optional

		shape {
		  dim {
		    dim_param: "batch_size"
                    dim_value: 32
		  }
		  dim {
                    dim_param: "channel"
		    dim_value: 3
		  }
		  dim {
                    dim_param: "height"
		    dim_value: 224
		  }
		  dim {
                    dim_param: "weight"
		    dim_value: 224
		  }
		}

            # we can access them like this
            # tensor_dim = model_input.type.tensor_type.shape.dim
            # print(tensor_dim[0].dim_param)
            # print(tensor_dim[0].dim_param)
            # print(tensor_dim[x].dim_param)
            # print(tensor_dim[x].dim_value)
            '''
            count += 1
            if count == nums:
                break

    def set_model_input_batch_size(self, index=0, name=None, batch_size=8):
        model_input = None
        if name is not None:
            # get model input by its name
            for mi in self.model.graph.input:
                if mi.name == name:
                    model_input = mi
        else:
            model_input = self.model.graph.input[index]

        if model_input:
            model_input = self.model.graph.input[index]
            tensor_dim = model_input.type.tensor_type.shape.dim
            tensor_dim[0].ClearField("dim_param")
            tensor_dim[0].dim_value = batch_size
        else:
            print("get model input error, check your index or name")

    def set_model_input_shape(self, index=0, name=None, shape=None):
        model_input = None
        if name is not None:
            # get model input by its name
            for mi in self.model.graph.input:
                if mi.name == name:
                    model_input = mi
        else:
            model_input = self.model.graph.input[index]

        if model_input:
            if shape is not None:
                model_input = self.model.graph.input[index]
                tensor_shape_proto = model_input.type.tensor_type.shape
                tensor_shape_proto.ClearField("dim")
                tensor_shape_proto.dim.extend([])
                for d in shape:
                    dim = tensor_shape_proto.dim.add()
                    dim.dim_value = d
            else:
                print("input shape must be set")
        else:
            print("get model input error, check your index or name")

    def get_node_by_name(self, name):
        for node in self.model.graph.node:
            if node.name == name:
                return node

    def get_nodes_by_optype(self, typename):
        nodes = []
        for node in self.model.graph.node:
            if node.op_type == typename:
                nodes.append(node)
        return nodes
    
    def fix_loop_body(self):
        while True:
            modify = False
            for node in self.model.graph.node:
                if node.op_type == 'convert_gradient_to_tensor_HBc3xYw22Mw':
                    print('find convert_gradient_to_tensor_HBc3xYw22Mw')
                    modify = True
                    bfs_remove(self.model.graph, node.name, node.name, connect=True)
            if not modify:
                break

        print('fixing loop body...')
        for node in self.model.graph.node:
            if node.op_type == 'Loop' :
                graph = node.attribute[0].g
                export_graph(graph, node.name.split('/')[-1]+'.onnx')
                print(len(graph.node))
                while True:
                    modify = False
                    for node in graph.node:
                        if node.op_type == 'convert_gradient_to_tensor_HBc3xYw22Mw':
                            print('find convert_gradient_to_tensor_HBc3xYw22Mw')
                            modify = True
                            bfs_remove(graph, node.name, node.name, connect=True)
                    if not modify:
                        break

                cnt = 0
                for n in graph.node:
                    if n.op_type == 'AddV2':
                        cnt += 1
                        n.op_type = 'Add'
                        n.domain = ''
                print(cnt," AddV2 nodes have been fixed")
                fix_IsFinite(graph)
                id = 0
                for i in range(len(graph.node)):
                    if graph.node[i].op_type == 'TopK':
                        topk = graph.node[i]
                        x = topk.output[1]
                        topk.output[1] = x+'before_cast'
                        
                        
                        id += 1
                        cast_node = onnx.helper.make_node('Cast'
                                                    ,name='cast_topk'+str(id)
                                                    ,inputs=[x+'before_cast']
                                                    ,outputs=[x]
                                                    )
                        cast_node.attribute.insert(0, onnx.helper.make_attribute('to', onnx.TensorProto.INT32, 'to'))
                        graph.node.insert(i+1, cast_node)
          
                        
    def able_run(self):
        nodes_to_remove = []
        
        for node in self.model.graph.node:
            if node.name == 'gec_ged_model_revised_gedloss_1/symbol_modality_28996_512/parallel_0/symbol_modality_28996_512/shared/Squeeze':
                node.input[0] = 'inputs:0'
            elif node.name == 'gec_ged_model_revised_gedloss_1/symbol_modality_28996_512/parallel_0/symbol_modality_28996_512/shared/mul_1':
                node.input[1] = 'gec_ged_model_revised_gedloss_1/symbol_modality_28996_512/parallel_0/symbol_modality_28996_512/shared/Cast:0'
            elif node.name == 'gec_ged_model_revised_gedloss_1/symbol_modality_28996_512/parallel_0/symbol_modality_28996_512/shared/ExpandDims':
                nodes_to_remove.append(node)
            




        bfs_remove(self.model.graph, 'gec_ged_model_revised_gedloss_1/ExpandDims', 'gec_ged_model_revised_gedloss_1/split')
        input_name = 'before_gather/squeeze:0'
        for node_index in get_node_topology(self.model.graph) :
            node = self.model.graph.node[node_index]
            if node.name == 'gec_ged_model_revised_gedloss_1/symbol_modality_28996_512/parallel_0/symbol_modality_28996_512/shared/GatherV2':
                x = node.input[1]
                node.input[1] = input_name
                sq_node = helper.make_node('Squeeze'
                                            ,name='insert_squeeze'
                                            ,inputs=[x]
                                            ,outputs=[input_name]
                                            )
                self.model.graph.node.insert(node_index, sq_node)
                break

        # for node in self.model.graph.node:
        #     if node.name == 'gec_ged_model_revised_gedloss_1/attention_bias_lower_triangle/attention_bias_local/Reshape':
        #         node.input[0] = 'gec_ged_model_revised_gedloss_1/attention_bias_lower_triangle/attention_bias_local/ones:0'
        #     elif node.op_type == 'MatrixBandPart':
        #         nodes_to_remove.append(node)

        remove_node_list(self.model.graph, nodes_to_remove)



    def fix_Transpose2(self):
        print('fixing transpse2')
        cnt = 0
        for node in self.model.graph.node:
            if node.op_type == 'Transpose2':
                concat_node = get_node_from_output_name_and_op_type(self.model.graph, node.input[1], 'Concat')
                if concat_node is not None:
                    remove_node(self.model.graph, concat_node)
                    print('concat_node has been removed')
                cnt += 1
                print('perm: ',node.input[1])
                #node.op_type = 'Transpose'
                #self.set_node_attribute(node, 'perm', [0,1,2])
                node.input.remove(node.input[1])
                
                self.remove_node(node)

        print(cnt," Transpose2 nodes have been fixed")

    def fix_AddV2(self):
        cnt = 0
        for node in self.model.graph.node:
            if node.op_type == 'AddV2':
                cnt += 1
                node.op_type = 'Add'
                node.domain = ''
        print(cnt," AddV2 nodes have been fixed")

    def fix_MatrixBandPart(self):
        cnt = 0
        for node in self.model.graph.node:
            if node.op_type == 'MatrixBandPart':
                cnt += 1
                node.domain = ''

                node.input.remove(node.input[1])
                node.input.remove(node.input[1])
                self.set_node_attribute(node, 'num_lower', -1)
                self.set_node_attribute(node, 'num_upper', 0)


        print(cnt," MatrixBandPart nodes have been fixed")
        get_node_topology(self.model.graph)

  
    def remove_pad(self):
        cnt = 0
        nodes_to_remove = []
        removed_nodes = set()
        for node in self.model.graph.node:
            if node.op_type == 'Pad':
                cnt += 1
                node.domain = ''
                reshape_node = get_node_from_input_name_and_op_type(self.model.graph, node.output[0], 'Reshape')
                if reshape_node is None:
                    print('can not found reshape_node')
                    pdb.set_trace()
                for n in self.model.graph.node:
                    if node_eq(n, reshape_node):
                        n.input[0] = node.input[0]
                if node.name not in removed_nodes :
                    nodes_to_remove.append(node)
                    removed_nodes.add(node.name)
                next_node = get_node_from_output_name_and_op_type(self.model.graph, node.input[1], 'Reshape')
                while True :
                    flag = False
                    if next_node.name not in removed_nodes :
                        nodes_to_remove.append(next_node)
                        removed_nodes.add(next_node.name)
                    for input_name in next_node.input :
                        if get_node_from_output_name(self.model.graph, input_name) is not None:
                            flag = True
                            next_node = get_node_from_output_name(self.model.graph, input_name)
                            break
                    if not flag :
                        break

        remove_node_list(self.model.graph, nodes_to_remove)
        print(cnt," Pad nodes have been fixed")

    def fix_ScatterNd(self) :
        op_index = 0
        while True :
            modify = False
            for node_index in get_node_topology(self.model.graph) :
                ScatterNd_node = self.model.graph.node[node_index]
                if ScatterNd_node.op_type != 'ScatterNd' :
                    continue
            
                op_index += 1
                print('find ScatterNd'+str(op_index))
                data_name = 'ConstantOfShape/before_ScatterND'+str(op_index)+':0'
                indice_name = ScatterNd_node.input[0]
                update_name = ScatterNd_node.input[1]
                shape_name = ScatterNd_node.input[2]
                ScatterNd_node.op_type = 'ScatterND'
                ScatterNd_node.input[0] = data_name
                ScatterNd_node.input[1] = indice_name
                ScatterNd_node.input[2] = update_name
                ScatterNd_node.domain = ''

                cast_n = get_node_from_output_name_and_op_type(self.model.graph, indice_name, 'Cast')
                for n in self.model.graph.node :
                    if node_eq(n, cast_n) :
                        self.set_node_attribute(n, 'to', 7)
                        break

                


                constant_node = onnx.helper.make_node('ConstantOfShape'
                                                    ,name='ConstantOfShape/before_ScatterND'+str(op_index)
                                                    ,inputs=[shape_name+'_cast']
                                                    ,outputs=[data_name]
                                                    )

                cast_node = onnx.helper.make_node('Cast'
                                                    , name = 'ConstantOfShape/cast'+str(op_index)
                                                    , inputs = [shape_name]
                                                    , outputs = [shape_name+'_cast']
                                                    )
                cast_node.attribute.insert(0, onnx.helper.make_attribute('to', onnx.TensorProto.INT64, 'to'))
                 
                self.model.graph.node.insert(node_index, constant_node)
                self.model.graph.node.insert(node_index, cast_node)
                modify = True
                break
            if not modify :
                break
                
    







    def get_weight_by_name(self, name):
        for weight in self.model.graph.initializer:
            if weight.name == name:
                return weight

    def set_weight_by_name(self, name, data_numpy=None, all_ones=False, all_zeros=False):
        weight = self.get_weight_by_name(name)
        self.set_weight(weight, data_numpy, all_ones, all_zeros)

    def remove_node_by_name(self, name):
        target_node = self.get_node_by_name(name)
        self.remove_node(target_node)

    def remove_node(self, target_node):
        '''
            remove the node with only one input and only one output
        '''
        node_input = target_node.input[0]
        node_output = target_node.output[0]
        # set input of successor node to predecessor node of target node
        for node in self.model.graph.node:
            for i, n in enumerate(node.input):
                if n == node_output:
                    node.input[i] = node_input

        target_names = set(target_node.input) & set([weight.name for weight in self.model.graph.initializer])
        self.remove_weights(target_names)
        target_names.add(node_output)
        self.remove_inputs(target_names)
        self.remove_value_infos(target_names)
        self.model.graph.node.remove(target_node)

    def remove_weights(self, name_list):
        rm_list = []
        for weight in self.model.graph.initializer:
            if weight.name in name_list:
                rm_list.append(weight)
        for weight in rm_list:
            self.model.graph.initializer.remove(weight)

    def remove_inputs(self, name_list):
        rm_list = []
        for input_t in self.model.graph.input:
            if input_t.name in name_list:
                rm_list.append(input_t)
        for input_t in rm_list:
            self.model.graph.input.remove(input_t)

    def remove_value_infos(self, name_list):
        rm_list = []
        for value_info in self.model.graph.value_info:
            if value_info.name in name_list:
                rm_list.append(value_info)
        for value_info in rm_list:
            self.model.graph.value_info.remove(value_info)

    def set_weight(self, weight, data_numpy=None, all_ones=False, all_zeros=False):
        # NOTE: weight can be stroed in human readable fields(float_data, int32_data, string_data, ...)
        # as well as raw_data, if we set weight by raw_data, we must clear the fields above to make it effective
        # NOTE: data_type between numpy and TensorProto
        if data_numpy is not None:
            raw_shape = tuple([i for i in weight.dims])
            new_shape = np.shape(data_numpy)
            if weight.data_type == 8:
                # string data type is special, it requires to store data in string_data field
                # NOT the raw_data field
                print("Can NOT handle string data type right now...")
                exit()
                # weight.string_data = bytes(data_numpy, encoding = "utf8")
                # weight.ClearField("raw_data")
            if new_shape != raw_shape:
                print("Warning: the new weight shape is not consistent with original shape!")
                weight.dims[:] = list(new_shape)
                for model_input in self.model.graph.input:
                    if model_input.name == weight.name:
                        # copy from onnx.helper...
                        tensor_shape_proto = model_input.type.tensor_type.shape
                        tensor_shape_proto.ClearField("dim")
                        tensor_shape_proto.dim.extend([])
                        for d in new_shape:
                            dim = tensor_shape_proto.dim.add()
                            dim.dim_value = d

            weight.ClearField("float_data")
            weight.ClearField("int32_data")
            weight.ClearField("int64_data")
            weight.raw_data = data_numpy.tobytes()
        else:
            if all_ones:
                wr = numpy_helper.to_array(weight)
                wn = np.ones_like(wr)
            elif all_zeros:
                wr = numpy_helper.to_array(weight)
                wn = np.zeros_like(wr)
            else:
                print("You must give a data_numpy to set the weight, or set the all_ones/all_zeros flag.")
                exit()
            weight.ClearField("float_data")
            weight.ClearField("int32_data")
            weight.ClearField("int64_data")
            weight.raw_data = wn.tobytes()

    def set_node_attribute(self, target_node, attr_name, attr_value):
        flag = False
        for attr in target_node.attribute:
            if (attr.name == attr_name):
                if attr.type == 1:
                    attr.f = attr_value
                elif attr.type == 2:
                    attr.i = attr_value
                elif attr.type == 3:
                    attr.s = attr_value
                elif attr.type == 4:
                    attr.t = attr_value
                elif attr.type == 5:
                    attr.g = attr_value
                # NOTE: For repeated composite types, we should use something like
                # del attr.xxx[:]
                # attr.xxx.extend([n1, n2, n3])
                elif attr.type == 6:
                    attr.floats[:] = attr_value
                elif attr.type == 7:
                    attr.ints[:] = attr_value
                elif attr.type == 8:
                    attr.strings[:] = attr_value
                else:
                    print("unsupported attribute data type with attribute name")
                    return False
                flag = True

        if not flag:
            # attribute not in original node
            print("Warning: you are appending a new attribute to the node!")
            target_node.attribute.append(helper.make_attribute(attr_name, attr_value))
            flag = True
        return flag

    def chunk_at(self, target_node):
        r_nodes = [target_node]
        r_input_names = [input_n for input_n in target_node.input]
        r_count = len(r_nodes) + len(r_input_names)

        while True:
            for node in self.model.graph.node:
                # print("nn", node.output)
                if node in r_nodes:
                    continue
                for o in node.output:
                    if o in r_input_names:
                        r_nodes.append(node)
                        r_input_names.extend([input_n for input_n in node.input])
                        continue
            n_count = len(r_nodes) + len(r_input_names)
            if n_count == r_count:
                break
            r_count = n_count

        print("debug r count", r_count)

        d_nodes = []
        d_inputs = []
        d_weights = []
        d_value_infos = []
        for node in self.model.graph.node:
            if node not in r_nodes:
                d_nodes.append(node)
        for model_input in self.model.graph.input:
            if model_input.name not in r_input_names:
                d_inputs.append(model_input)
        for weight in self.model.graph.initializer:
            if weight.name not in r_input_names:
                d_weights.append(weight)
        for value_info in self.model.graph.value_info:
            if value_info.name not in r_input_names:
                d_values.append(value_info)
        for node in d_nodes:
            self.model.graph.node.remove(node)
        for model_input in d_inputs:
            self.model.graph.input.remove(model_input)
        for weight in d_weights:
            self.model.graph.initializer.remove(weight)
        for value_info in d_value_infos:
            self.model.graph.value_info.remove(value_info)

        target_node.output[0] = self.model.graph.output[0].name
        # remove other outputs if model has multi-output
        d_outputs = []
        for i, output in enumerate(self.model.graph.output):
            if i != 0 :
                d_outputs.append(output)
        for output in d_outputs:
            self.model.graph.output.remove(output)

    def insert_flatten_before(self, target_node):
        # get target_node inputs
        node_input = target_node.input[0]
        # create new node
        node_name = "flatten_test"
        flatten_node = helper.make_node('Flatten', inputs=[node_input], outputs=[node_name], name=node_name)
        # set target_node inputs to new node outputs
        target_node.input[0] = node_name
        for target_node_index, _target_node in enumerate(self.model.graph.node):
            if _target_node == target_node:
                self.model.graph.node.insert(target_node_index, flatten_node)
                break

    def insert_op_before(self, node_name, target_node, input_idx=0, *args, **kwargs):
        '''
        op_name
        weight_dict
        attr_dict
        ......

        NOTE:
        you must ensure the output shape match the input shape of target_node
        '''
        # get target_node inputs
        node_input = target_node.input[input_idx]
        weight_input = []
        weight_input_vi = []
        weight_initializer = []
        if "weight_dict" in kwargs:
            for weight_name, weight_numpy in kwargs["weight_dict"].items():
                weight_input.append(weight_name)
                weight_input_vi.append(
                        helper.make_tensor_value_info(
                            name=weight_name,
                            elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_numpy.dtype],
                            shape=weight_numpy.shape
                        )
                )
                weight_initializer.append(
                    helper.make_tensor(
                            name=weight_name,
                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_numpy.dtype],
                            dims=weight_numpy.shape,
                            vals=weight_numpy.tobytes(),
                            raw=True
                    )
                )
        # create new node
        new_op_node = helper.make_node(
                                kwargs["op_name"],
                                inputs=[node_input, *weight_input],
                                outputs=[node_name],
                                name=node_name,
                                **kwargs["attr_dict"]
                            )
        # set target_node input to new node outputs
        target_node.input[input_idx] = node_name
        # TODO: change other nodes input into the new node?
        # iterator all the nodes in the graph and find
        # which node's input equals the original target_node input
        # ...
        # add new node and weight input into the graph
        for target_node_index, _target_node in enumerate(self.model.graph.node):
            if _target_node == target_node:
                self.model.graph.node.insert(target_node_index, new_op_node)
                break
        self.model.graph.input.extend(weight_input_vi)
        self.model.graph.initializer.extend(weight_initializer)

    def add_extra_output(self, target_node, output_name):
        extra_output = helper.make_empty_tensor_value_info(output_name)
        '''
            # NOTE
            # if we know the value type and shape, we can alse use this
	    def make_tensor_value_info(
		    name,  # type: Text
		    elem_type,  # type: int
		    shape,  # type: Optional[Sequence[Union[Text, int]]]
		    doc_string="",  # type: Text
		    shape_denotation=None,  # type: Optional[List[Text]]
	    ):
        '''
        target_output = target_node.output[0]
        identity_node = helper.make_node('Identity', inputs=[target_output], outputs=[output_name], name=output_name)
        self.model.graph.node.append(identity_node)
        self.model.graph.output.append(extra_output)


