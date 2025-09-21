from .utils import CSG_var_to_code_tokens_pos, generate_index_to_code_token_dict


def generate_CSG_df_subgraph_java(root_node,index_to_code,states,code,new_tokens):
    def_statement = ['variable_declarator']
    assignment=['assignment_expression']
    increment_statement=['update_expression']
    if_statement=['if_statement']
    for_statement=['for_statement']
    while_statement=['while_statement']
    special_situaion = ['array_access', 'method_invocation']
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
            indexs = CSG_var_to_code_tokens_pos(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                csg_df.append((code, idx, 'declaration', [], []))
                states[code] = [idx]
            return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
        else:
            name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code)
            value_indexs = CSG_var_to_code_tokens_pos(value, index_to_code)
            temp, states, new_tokens = generate_CSG_df_subgraph_java(value, index_to_code, states, code, new_tokens)
            csg_df += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        right_nodes=root_node.child_by_field_name('right')
        csg_df=[]
        temp,states,new_tokens=generate_CSG_df_subgraph_java(right_nodes,index_to_code,states,code, new_tokens)
        csg_df+=temp
        name_indexs=CSG_var_to_code_tokens_pos(left_nodes,index_to_code)
        value_indexs=CSG_var_to_code_tokens_pos(right_nodes,index_to_code)
        for index1 in name_indexs:
            idx1,code1=index_to_code[index1]
            for index2 in value_indexs:
                idx2,code2=index_to_code[index2]
                csg_df.append((code1,idx1,'comes_from',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in increment_statement:
        csg_df=[]
        indexs=CSG_var_to_code_tokens_pos(root_node,index_to_code)
        for index1 in indexs:
            idx1,code1=index_to_code[index1]
            for index2 in indexs:
                idx2,code2=index_to_code[index2]
                csg_df.append((code1,idx1,'comes_from',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in if_statement:
        csg_df=[]
        current_states=states.copy()
        others_states=[]
        flag = False
        tag = False
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states, new_tokens = generate_CSG_df_subgraph_java(child,index_to_code,current_states,code,new_tokens)
                csg_df += temp
            else:
                flag = True
                temp, new_states, new_tokens = generate_CSG_df_subgraph_java(child, index_to_code, states, code, new_tokens)
                csg_df += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
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
        for child in root_node.children:
            temp,states,new_tokens=generate_CSG_df_subgraph_java(child,index_to_code,states,code,new_tokens)
            csg_df+=temp
        for child in root_node.children:
            temp,states,new_tokens=generate_CSG_df_subgraph_java(child,index_to_code,states,code,new_tokens)
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
                temp,states,new_tokens=generate_CSG_df_subgraph_java(child,index_to_code,states,code,new_tokens)
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
        if root_node.type == 'array_access':
            csg_df = []
            name = root_node.children[0]
            size = root_node.children[2]
            name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code)
            size_indexs = CSG_var_to_code_tokens_pos(size, index_to_code)
            temp, states, new_tokens = generate_CSG_df_subgraph_java(size,index_to_code,states,code,new_tokens)
            csg_df += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in size_indexs:
                    idx2, code2 = index_to_code[index2]
                    csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
        elif root_node.type == 'method_invocation':
            if len(root_node.children) <= 2:
                csg_df = []
                for child in root_node.children:
                    temp, states, new_tokens = generate_CSG_df_subgraph_java(child, index_to_code, states, code,
                                                                             new_tokens)
                    csg_df += temp
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
            function_node = root_node.children[2]
            function_index = CSG_var_to_code_tokens_pos(function_node, index_to_code)
            function_name = generate_index_to_code_token_dict(function_index[0], code)
            input_func_list = ["read", "readLine", "nextLine", "nextInt", "nextDouble"]
            mem_func_list = ["arraycopy"]
            if function_name in input_func_list:
                csg_df = []
                argument_list = root_node.children[3]
                argument_indexs = CSG_var_to_code_tokens_pos(argument_list, index_to_code)
                temp, states, new_tokens = generate_CSG_df_subgraph_java(argument_list, index_to_code, states, code, new_tokens)
                csg_df += temp
                csg_df.append(('input',len(index_to_code),'declaration',[],[]))
                for index1 in argument_indexs:
                    idx1, code1 = index_to_code[index1]
                    csg_df.append((code1, idx1, 'comes_from', ['input'], [len(index_to_code)+len(new_tokens)]))
                    states[code1] = [idx1]
                new_tokens.append('input')
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
            elif function_name in mem_func_list:
                csg_df = []
                argument_list = root_node.children[3]
                dst = argument_list.children[5]
                src = argument_list.children[1]
                temp, states, new_tokens = generate_CSG_df_subgraph_java(argument_list,index_to_code,states,code,new_tokens)
                csg_df += temp
                dst_indexs = CSG_var_to_code_tokens_pos(dst, index_to_code)
                src_indexs = CSG_var_to_code_tokens_pos(src, index_to_code)
                for index1 in dst_indexs:
                    idx1, code1 = index_to_code[index1]
                    for index2 in src_indexs:
                        idx2, code2 = index_to_code[index2]
                        csg_df.append((code1, len(index_to_code)+len(new_tokens), 'comes_from', [code2], [idx2]))
                        csg_df.append((code1, len(index_to_code)+len(new_tokens), 'comes_from', [code1], [idx1]))
                    states[code1] = [len(index_to_code)+len(new_tokens)]
                    new_tokens.append(code1)
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
            else:
                csg_df = []
                for child in root_node.children:
                    temp, states, new_tokens = generate_CSG_df_subgraph_java(child,index_to_code,states,code,new_tokens)
                    csg_df += temp
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
    else:
        csg_df=[]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_df_subgraph_java(child,index_to_code,states,code,new_tokens)
            csg_df += temp
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens


def generate_CSG_cf_subgraph_java(root_node):
    if_statement=['if_statement']
    for_statement=['for_statement']
    while_statement=['while_statement']
    if root_node.type in if_statement:
        csg_cf=[]
        flag_node = root_node.children[0]
        csg_cf.append((flag_node, 0, 'declaration', [], []))
        condition_node1 = root_node.children[1]
        condition_node2 = condition_node1
        if_node = root_node.children[2]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_java(child)
            csg_cf += temp
        csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
        csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
        csg_cf.append((if_node, 2, 'determined_by', [condition_node1], []))
        if len(root_node.children) == 5:
            else_node = root_node.children[4]
            csg_cf.append((else_node, 2, 'determined_by', [condition_node2], []))
        return csg_cf
    elif root_node.type in for_statement:
        csg_cf=[]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_java(child)
            csg_cf += temp
        flag_node = root_node.children[0]
        csg_cf.append((flag_node, 0, 'declaration', [], []))
        condition_node1 = root_node.children[0]
        condition_node2 = root_node.children[0]
        for child in root_node.children:
            if child.type == 'binary_expression':
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
        body_node = root_node.children[2]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_java(child)
            csg_cf += temp
        csg_cf.append((flag_node, 0, 'declaration', [], []))
        csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
        csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
        csg_cf.append((body_node, 2, 'determined_by', [condition_node1], []))
        return csg_cf
    else:
        csg_cf=[]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_java(child)
            csg_cf += temp
        return csg_cf

        
def generate_CSG_vul_subgraph_java(root_node, index_to_code, states, code, new_tokens):
    def_statement = ['variable_declarator']
    function_name = ['method_declaration']
    for_statement=['for_statement']
    while_statement=['while_statement']
    api_relation = ['method_invocation']
    array_relation = ['array_access']
    num_literal = ['decimal_integer_literal']
    states=states.copy()
    if root_node.type in function_name:
        func_name = root_node.children[2]
        if func_name.type != 'identifier':
            return [], states, new_tokens
        idx1, code1 = index_to_code[(func_name.start_point, func_name.end_point)]
        states[code1] = [idx1]
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        return csg_vul, states, new_tokens
    elif root_node.type in num_literal:
        idx, code_token = index_to_code[(root_node.start_point, root_node.end_point)]
        csg_vul = []
        if code_token.isdigit():
            if int(code_token) > 10000:
                csg_vul = [(root_node, 0, 'check', ['vul'], [])]
        return csg_vul, states, new_tokens
    elif root_node.type in array_relation:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        csg_vul.append((root_node.children[2], 0, 'check', ['vul'], []))
        csg_vul.append((root_node.children[0], 1, 'check', [root_node.children[2]], []))
        csg_vul.append((root_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in def_statement:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        identifier = root_node.children[0]
        if identifier.type == 'identifier':
            csg_vul.append((root_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in for_statement:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        for child in root_node.children:
            if child.type == 'binary_expression':
                csg_vul.append((child, 0, 'check', ['vul'], []))
                break
        return csg_vul, states, new_tokens
    elif root_node.type in while_statement:
        csg_vul = []
        condition_node = root_node.children[1]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        csg_vul.append((condition_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in api_relation:
        if len(root_node.children) <= 2:
            csg_vul = []
            for child in root_node.children:
                temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code,
                                                                         new_tokens)
                csg_vul += temp
            return csg_vul, states, new_tokens
        function_node = root_node.children[2]
        function_index = CSG_var_to_code_tokens_pos(function_node, index_to_code)
        function_name = generate_index_to_code_token_dict(function_index[0], code)
        input_func_list = ["read", "readLine", "nextLine", "nextInt", "nextDouble"]
        mem_func_list = ["arraycopy"]
        if function_name in input_func_list:
            csg_vul = []
            argument_list = root_node.children[3]
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in mem_func_list:
            csg_vul = []
            argument_list = root_node.children[3]
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            csg_vul.append((argument_list, 0, '<', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in states:
            csg_vul = []
            argument_list = root_node.children[3]
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            csg_vul.append((root_node, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        else:
            csg_vul = []
            for child in root_node.children:
                temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
                csg_vul += temp
            return csg_vul, states, new_tokens
    else:
        csg_vul=[]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_java(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        return csg_vul, states, new_tokens
