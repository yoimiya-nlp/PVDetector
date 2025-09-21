from .utils import CSG_var_to_code_tokens_pos, generate_index_to_code_token_dict


def generate_CSG_df_subgraph_py(root_node,index_to_code,states,code,new_tokens):
    def_statement = ['default_parameter']
    assignment=['assignment', 'augmented_assignment', 'for_in_clause']
    if_statement=['if_statement']
    for_statement=['for_statement']
    while_statement=['while_statement']
    special_situaion = ['subscript', 'call']
    states=states.copy()

    if (len(root_node.children)==0 or root_node.type.find('string') != -1) and root_node.type!='comment':

        idx,code=index_to_code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states,new_tokens
        elif code in states:
            return [(code,idx,'equals_to',[code],states[code].copy())],states,new_tokens
        else:
            if root_node.type=='identifier':  
                states[code]=[idx]
            return [(code,idx,'declaration',[],[])],states,new_tokens
    elif root_node.type in def_statement:
        csg_df = []
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        if value is None:
            name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code)
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                csg_df.append((code1, idx1, 'declaration', [], []))
                states[code1] = [idx1]
            return sorted(csg_df, key=lambda x: x[1]), states, new_tokens

        name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code)   
        value_indexs = CSG_var_to_code_tokens_pos(value, index_to_code)
        temp, states, new_tokens = generate_CSG_df_subgraph_py(value, index_to_code, states, code, new_tokens)
        csg_df += temp

        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))

            states[code1] = [idx1]
        return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
    elif root_node.type in assignment:
        if root_node.type == 'for_in_clause':
            right_nodes = [root_node.children[-1]]
            left_nodes = [root_node.child_by_field_name('left')]
        else:
            if root_node.child_by_field_name('right') is None:
                return [], states, new_tokens
            left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
            right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name('left')]
                right_nodes = [root_node.child_by_field_name('right')]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name('left')]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name('right')]

        csg_df=[]
        for node in right_nodes:
            temp,states,new_tokens=generate_CSG_df_subgraph_py(node,index_to_code,states,code,new_tokens)
            csg_df+=temp

        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index=CSG_var_to_code_tokens_pos(left_node,index_to_code)
            right_tokens_index=CSG_var_to_code_tokens_pos(right_node,index_to_code)
            temp = []
            for index1 in left_tokens_index:
                idx1, code1 = index_to_code[index1]
                temp.append((code1, idx1, 'comes_from', [index_to_code[x][1] for x in right_tokens_index],
                             [index_to_code[x][0] for x in right_tokens_index]))
                states[code1] = [idx1]
            csg_df += temp
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in if_statement:
        csg_df=[]
        current_states=states.copy()
        others_states=[]
        tag=False
        for child in root_node.children:
            if 'else' in root_node.type:
                tag = True
            if child.type not in ['elif_clause','else_clause']:
                temp, current_states,new_tokens = \
                    generate_CSG_df_subgraph_py(child,index_to_code,current_states,code,new_tokens)
                csg_df += temp
            else: 
                temp,new_states,new_tokens=generate_CSG_df_subgraph_py(child,index_to_code,states,code,new_tokens)
                csg_df+=temp
        others_states.append(current_states)
        if tag is False:    
            others_states.append(states)
        new_states={}
        for dic in others_states: 
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key]))) 
        return sorted(csg_df,key=lambda x:x[1]), new_states, new_tokens
    elif root_node.type in for_statement:
        csg_df=[]
        for i in range(2):
            right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
            left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name('left')]
                right_nodes = [root_node.child_by_field_name('right')]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name('left')]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name('right')]

            for node in right_nodes:
                temp,states,new_tokens=generate_CSG_df_subgraph_py(node,index_to_code,states,code,new_tokens)
                csg_df+=temp
            for left_node, right_node in zip(left_nodes, right_nodes):
                left_tokens_index=CSG_var_to_code_tokens_pos(left_node,index_to_code)
                right_tokens_index=CSG_var_to_code_tokens_pos(right_node,index_to_code)
                temp = []
                for index1 in left_tokens_index:
                    idx1, code1 = index_to_code[index1]
                    temp.append((code1, idx1, 'comes_from', [index_to_code[x][1] for x in right_tokens_index],
                                 [index_to_code[x][0] for x in right_tokens_index]))
                    states[code1] = [idx1]
                csg_df += temp
            if root_node.children[-1].type=="block":
                temp,states,new_tokens=\
                    generate_CSG_df_subgraph_py(root_node.children[-1],index_to_code,states,code,new_tokens)
                csg_df+=temp

        dic={}
        for x in csg_df: 
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        csg_df=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in while_statement:
        csg_df=[]
        for i in range(2):
            for child in root_node.children:
                temp,states,new_tokens=generate_CSG_df_subgraph_py(child,index_to_code,states,code,new_tokens)
                csg_df+=temp
        dic={}
        for x in csg_df:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        csg_df=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in special_situaion:
        if root_node.type == 'subscript':
            csg_df = []
            name = root_node.children[0]
            index = root_node.children[2]

            name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code)
            size_indexs = CSG_var_to_code_tokens_pos(index, index_to_code)
            temp, states, new_tokens = generate_CSG_df_subgraph_py(index,index_to_code,states,code,new_tokens)
            csg_df += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in size_indexs:
                    idx2, code2 = index_to_code[index2]
                    csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
        elif root_node.type == 'call':
            function_node = root_node.children[0]
            function_index = CSG_var_to_code_tokens_pos(function_node, index_to_code)
            function_name = generate_index_to_code_token_dict(function_index[0], code)
            input_func_list = ["input", "open", "sys.argv"]

            if function_name in input_func_list:
                csg_df = []
                argument_list = root_node.children[1]
                argument_indexs = CSG_var_to_code_tokens_pos(argument_list, index_to_code)
                temp, states, new_tokens = generate_CSG_df_subgraph_py(argument_list, index_to_code, states, code, new_tokens)
                csg_df += temp
                csg_df.append(('input',len(index_to_code),'declaration',[],[]))
                for index1 in argument_indexs:
                    idx1, code1 = index_to_code[index1]
                    csg_df.append((code1, idx1, 'comes_from', ['input'], [len(index_to_code)+len(new_tokens)]))
                    states[code1] = [idx1]
                new_tokens.append('input')
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
            else:
                csg_df = []
                for child in root_node.children:
                    temp, states, new_tokens = generate_CSG_df_subgraph_py(child,index_to_code,states,code,new_tokens)
                    csg_df += temp
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
    else:
        csg_df=[]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_df_subgraph_py(child,index_to_code,states,code,new_tokens)
            csg_df += temp
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens


def generate_CSG_cf_subgraph_py(root_node):
    if_statement=['if_statement']
    for_statement=['for_statement']
    while_statement=['while_statement']
    
    if root_node.type in if_statement:
        csg_cf=[]

        flag_node = root_node.children[0]
        csg_cf.append((flag_node, 0, 'declaration', [], []))
        if_condition = root_node.children[1]
        if_node = root_node.children[3]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_py(child)
            csg_cf += temp

        csg_cf.append((if_condition, 1, 'determined_by', [flag_node], []))
        csg_cf.append((if_node, 1, 'determined_by', [if_condition], []))
        i = 0
        for child in root_node.children:
            if child.type == 'elif_clause':
                elif_condition = child.children[1]
                elif_node = child.children[3]
                csg_cf.append((elif_condition, 3+i, 'determined_by', [flag_node], []))
                csg_cf.append((elif_node, 1, 'determined_by', [elif_condition], []))
                i = i + 2
                continue
            if child.type == 'else_clause':
                else_condition = child.children[0]
                else_node = child.children[2]
                csg_cf.append((else_condition, 3+i, 'determined_by', [flag_node], []))
                csg_cf.append((else_node, 1, 'determined_by', [else_condition], []))

        return csg_cf
    elif root_node.type in for_statement:
        csg_cf=[]

        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_py(child)
            csg_cf += temp

        flag_node = root_node.children[0]
        csg_cf.append((flag_node, 0, 'declaration', [], []))
        condition_node1 = root_node.children[0]
        condition_node2 = root_node.children[0]
        for child in root_node.children:
            if child.type == 'in':
                condition_node1 = child
                condition_node2 = child
                csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
                csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
            elif child.type == 'block':
                csg_cf.append((child, 2, 'determined_by', [condition_node1], []))

        return csg_cf
    elif root_node.type in while_statement:
        csg_cf=[]

        flag_node = root_node.children[0]
        condition_node1 = root_node.children[1]
        condition_node2 = condition_node1
        body_node = root_node.children[3]

        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_py(child)
            csg_cf += temp

        csg_cf.append((flag_node, 0, 'declaration', [], []))
        csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
        csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
        csg_cf.append((body_node, 2, 'determined_by', [condition_node1], []))

        return csg_cf
    else:
        csg_cf=[]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_py(child)
            csg_cf += temp
        return csg_cf


def generate_CSG_vul_subgraph_py(root_node, index_to_code, states, code, new_tokens):
    for_statement=['for_statement']
    while_statement=['while_statement']
    api_relation = ['call']
    array_relation = ['subscript']
    num_literal = ['integer']
    states=states.copy()
    
    if root_node.type in num_literal:
        idx, code_token = index_to_code[(root_node.start_point, root_node.end_point)]
        csg_vul = []
        if code_token.isdigit():
            if int(code_token) > 10000:
                csg_vul = [(root_node, 0, 'check', ['vul'], [])]
        return csg_vul, states, new_tokens
    elif root_node.type in array_relation:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_py(child, index_to_code, states, code, new_tokens)
            csg_vul += temp

        csg_vul.append((root_node, 0, 'check', ['vul'], []))
        list_node = root_node.children[0]
        index_node = root_node.children[2]
        csg_vul.append((list_node, 0, 'check', ['vul'], []))
        csg_vul.append((index_node, 1, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in for_statement:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_py(child, index_to_code, states, code, new_tokens)
            csg_vul += temp

        for child in root_node.children:
            if child.type == 'in':
                csg_vul.append((child, 0, 'check', ['vul'], []))
                break
        return csg_vul, states, new_tokens
    elif root_node.type in while_statement:
        csg_vul = []
        condition_node = root_node.children[1]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_py(child, index_to_code, states, code, new_tokens)
            csg_vul += temp

        csg_vul.append((condition_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in api_relation:
        function_node = root_node.children[0]
        function_index = CSG_var_to_code_tokens_pos(function_node, index_to_code)
        function_name = generate_index_to_code_token_dict(function_index[0], code)
        input_func_list = ["input", "open", "sys.argv"]
        if function_name in input_func_list:
            csg_vul = []
            argument_list = root_node.children[1]
            temp, states, new_tokens = generate_CSG_vul_subgraph_py(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp

            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in states:
            csg_vul = []
            argument_list = root_node.children[1]
            temp, states, new_tokens = generate_CSG_vul_subgraph_py(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp

            csg_vul.append((root_node, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        else:
            csg_vul = []
            for child in root_node.children:
                temp, states, new_tokens = generate_CSG_vul_subgraph_py(child, index_to_code, states, code, new_tokens)
                csg_vul += temp
            return csg_vul, states, new_tokens
    else:
        csg_vul=[]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_py(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        return csg_vul, states, new_tokens
