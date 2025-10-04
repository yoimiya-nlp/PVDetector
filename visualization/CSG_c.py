from utils import CSG_var_to_code_tokens_pos, generate_index_to_code_token_dict


def generate_CSG_df_subgraph_c(root_node,index_to_code,states,code,new_tokens):
    def_statement = ['init_declarator']
    assignment=['assignment_expression']
    increment_statement=['update_expression']
    if_statement=['if_statement']
    for_statement=['for_statement']
    switch_statement=['switch_statement']
    while_statement=['while_statement']
    do_first_statement=['do_statement']
    special_situaion = ['array_declarator', 'call_expression']
    states=states.copy()
    # CSG_df: child, child idx, comefrom, [father], [father idx]
    # index_to_code[index] = (idx,code)
    if (len(root_node.children)==0 or root_node.type.find('string') != -1) and root_node.type!='comment':
        # leaf node
        idx,code = index_to_code[(root_node.start_point,root_node.end_point)]
        if root_node.type == code:
            return [], states,new_tokens
        elif code in states:
            return [(code,idx, 'equals_to', [code], states[code].copy())], states, new_tokens
        else:
            if root_node.type == 'identifier':      # first time encounter
                states[code] = [idx]
            return [(code, idx, 'declaration', [], [])], states, new_tokens
    elif root_node.type in def_statement:           # only decide assignment
        csg_df = []
        name = root_node.children[0]
        value = root_node.children[2]

        name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code) 
        value_indexs = CSG_var_to_code_tokens_pos(value, index_to_code)
        temp, states, new_tokens = generate_CSG_df_subgraph_c(value, index_to_code, states, code, new_tokens)
        csg_df += temp
        
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))
                # in-statement data flow
            states[code1] = [idx1]
        return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')        # children[0]
        right_nodes = root_node.child_by_field_name('right')      # children[2]
        csg_df = []
        temp,states, new_tokens = generate_CSG_df_subgraph_c(right_nodes,index_to_code,states,code, new_tokens)
        csg_df += temp

        name_indexs = CSG_var_to_code_tokens_pos(left_nodes,index_to_code)
        value_indexs = CSG_var_to_code_tokens_pos(right_nodes,index_to_code)

        for index1 in name_indexs:
            idx1,code1=index_to_code[index1]
            for index2 in value_indexs:
                idx2,code2=index_to_code[index2]
                csg_df.append((code1,idx1,'comes_from',[code2],[idx2]))
            states[code1]=[idx1]   
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in increment_statement:
        # a++
        csg_df = []
        indexs = CSG_var_to_code_tokens_pos(root_node.children[0],index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(csg_df, key=lambda x:x[1]), states, new_tokens
    elif root_node.type in if_statement:
        csg_df = []
        current_states = states.copy()
        others_states = []

        haveelse_tag = False
        for child in root_node.children:
            if child.type == 'if':
                continue
            if child.type == 'parenthesized_expression':
                temp, states,new_tokens = generate_CSG_df_subgraph_c(child, index_to_code, states, code,new_tokens)
                current_states = states.copy()          # before if statement
                csg_df += temp
                continue
            if child.type == 'else_clause':             # else clause
                haveelse_tag = True
                temp, new_states, new_tokens = generate_CSG_df_subgraph_c(child, index_to_code, current_states, code, new_tokens)
                csg_df += temp
                others_states.append(new_states)
            else:                                       # if clause
                temp,states,new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,states,code,new_tokens)
                csg_df += temp
        others_states.append(states)

        if haveelse_tag is False:           # no else
            others_states.append(current_states)
        new_states={}
        for dic in others_states:           # merge
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))      # duplicate
        return sorted(csg_df,key=lambda x:x[1]), new_states, new_tokens
    elif root_node.type in for_statement:
        csg_df = []
        for child in root_node.children:            # because of loop
            temp, states, new_tokens = generate_CSG_df_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_df += temp
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_df_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_df += temp

        dic = {}
        for x in csg_df:                               # merge same child node
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        csg_df=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in switch_statement:
        csg_df = []
        current_states = states.copy()
        others_states = []
        for child in root_node.children:
            if child.type == 'switch':
                continue
            if child.type == 'parenthesized_expression':
                temp, states, new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,states,code,new_tokens)
                current_states = states.copy()  # before switch statement
                csg_df += temp
            else:                               # compound_statement clause
                for child_child in child.children:
                      if child_child.type == 'case_statement':
                          temp, new_states, new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,current_states,code,new_tokens)
                          csg_df += temp
                          others_states.append(new_states)

        others_states.append(current_states)
        new_states = {}
        for dic in others_states:  # merge
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(csg_df, key=lambda x:x[1]), new_states, new_tokens
    elif root_node.type in while_statement:  
        csg_df = []
        for i in range(2):
            for child in root_node.children:
                temp,states,new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,states,code,new_tokens)
                csg_df += temp
        dic={}
        for x in csg_df:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        csg_df=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens
    elif root_node.type in do_first_statement:
        csg_df = []
        for i in range(2):
            for child in root_node.children:
                temp, states, new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,states,code,new_tokens)
                csg_df += temp
        dic = {}
        for x in csg_df:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        csg_df = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(csg_df, key=lambda x:x[1]), states, new_tokens
    elif root_node.type in special_situaion:
        if root_node.type == 'array_declarator':
            csg_df = []
            name = root_node.children[0]
            size = root_node.children[2]

            name_indexs = CSG_var_to_code_tokens_pos(name, index_to_code) 
            size_indexs = CSG_var_to_code_tokens_pos(size, index_to_code)
            temp, states, new_tokens = generate_CSG_df_subgraph_c(size,index_to_code,states,code,new_tokens)
            csg_df += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in size_indexs:
                    idx2, code2 = index_to_code[index2]
                    csg_df.append((code1, idx1, 'comes_from', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
        elif root_node.type == 'call_expression':
            function_node = root_node.children[0]
            function_index = CSG_var_to_code_tokens_pos(function_node, index_to_code)
            function_name = generate_index_to_code_token_dict(function_index[0], code)
            input_func_list = ["gets", "scanf", "fscanf", "getchar", "fgetc", "fgets"]
            mem_func_list = ["memcpy", "strcpy", "strncpy", "strcat"]
            if function_name in input_func_list:
                csg_df = []
                argument_list = root_node.children[1]
                argument_indexs = CSG_var_to_code_tokens_pos(argument_list, index_to_code)
                temp, states, new_tokens = generate_CSG_df_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
                csg_df += temp
                csg_df.append(('input',len(index_to_code),'declaration',[],[]))
                n_input = len(index_to_code)+len(new_tokens)
                new_tokens.append('input')
                for index1 in argument_indexs:
                    idx1, code1 = index_to_code[index1]
                    csg_df.append((code1, len(index_to_code)+len(new_tokens), 'comes_from', ['input'], [n_input]))
                    csg_df.append((code1, len(index_to_code)+len(new_tokens), 'comes_from', [code1], [idx1]))
                    states[code1] = [len(index_to_code)+len(new_tokens)]
                    new_tokens.append(code1)
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
            elif function_name in mem_func_list:
                csg_df = []
                argument_list = root_node.children[1]
                dst = argument_list.children[1]
                src = argument_list.children[3]

                temp, states, new_tokens = generate_CSG_df_subgraph_c(argument_list,index_to_code,states,code,new_tokens)
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
                    temp, states, new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,states,code,new_tokens)
                    csg_df += temp
                return sorted(csg_df, key=lambda x: x[1]), states, new_tokens
    else:
        csg_df=[]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_df_subgraph_c(child,index_to_code,states,code,new_tokens)
            csg_df += temp
        return sorted(csg_df,key=lambda x:x[1]), states, new_tokens


def generate_CSG_cf_subgraph_c(root_node):
    if_statement=['if_statement']
    for_statement=['for_statement']
    switch_statement=['switch_statement']
    while_statement=['while_statement']
    do_first_statement=['do_statement']
    # CSG_cf: child，offset, determined_by, [father], [node_tokens_index]
    if root_node.type in if_statement:
        csg_cf=[]

        flag_node = root_node.children[0]
        csg_cf.append((flag_node, 0, 'declaration', [], []))
        condition_node1 = root_node.children[1]
        condition_node2 = condition_node1

        if_node = root_node.children[2]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_c(child)
            csg_cf += temp

        csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
        csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
        csg_cf.append((if_node, 2, 'determined_by', [condition_node1], []))

        if len(root_node.children) == 4:
            else_node = root_node.children[3]
            csg_cf.append((else_node, 2, 'determined_by', [condition_node2], []))

        return csg_cf
    elif root_node.type in for_statement:
        csg_cf=[]

        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_c(child)
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
            elif child.type == 'compound_statement':
                csg_cf.append((child, 2, 'determined_by', [condition_node1], []))

        return csg_cf
    elif root_node.type in switch_statement:
        csg_cf = []

        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_c(child)
            csg_cf += temp

        condition_node = root_node.children[1]
        csg_cf.append((condition_node, 0, 'declaration', [], []))
        body_node = root_node.children[2]
        for idx, child in enumerate(body_node.children):
            if child.type == 'case_statement':
                csg_cf.append((child, idx + 1, 'determined_by', [condition_node], []))

        return csg_cf
    elif root_node.type in while_statement:
        csg_cf = []

        flag_node = root_node.children[0]
        condition_node1 = root_node.children[1]
        condition_node2 = condition_node1
        body_node = root_node.children[2]

        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_c(child)
            csg_cf += temp

        csg_cf.append((flag_node, 0, 'declaration', [], []))
        csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
        csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
        csg_cf.append((body_node, 2, 'determined_by', [condition_node1], []))

        return csg_cf
    elif root_node.type in do_first_statement:
        csg_cf = []

        flag_node = root_node.children[2]
        condition_node1 = root_node.children[3]
        condition_node2 = condition_node1
        body_node = root_node.children[1]

        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_c(child)
            csg_cf += temp

        csg_cf.append((flag_node, 0, 'declaration', [], []))
        csg_cf.append((condition_node1, 1, 'determined_by', [flag_node], []))
        csg_cf.append((condition_node2, 2, 'determined_by', [flag_node], []))
        csg_cf.append((body_node, 2, 'determined_by', [condition_node1], []))

        return csg_cf
    else:
        csg_cf=[]
        for child in root_node.children:
            temp = generate_CSG_cf_subgraph_c(child)
            csg_cf += temp
        return csg_cf


def generate_CSG_vul_subgraph_c(root_node, index_to_code, states, code, new_tokens):
    def_statement = ['declaration']
    function_name = ['function_declarator']
    for_statement=['for_statement']
    while_statement=['while_statement']
    do_first_statement=['do_statement']
    api_relation = ['call_expression']
    array_relation = ['array_declarator', 'subscript_expression']
    pointer_relation = ['pointer_declarator', 'pointer_expression']
    num_literal = ['number_literal']
    int_type = ['primitive_type']
    null_pointer = ['null']
    states=states.copy()
    # CSG_vul: vul_element，offset, 'check', ['vul'], []
    if root_node.type in function_name:
        func_name = root_node.children[0]
        if func_name.type != 'identifier':
            return [], states, new_tokens
        idx1, code1 = index_to_code[(func_name.start_point, func_name.end_point)]
        states[code1] = [idx1]

        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        return csg_vul, states, new_tokens
    elif root_node.type in num_literal:
        idx, code_token = index_to_code[(root_node.start_point, root_node.end_point)]
        csg_vul = []
        if code_token.isdigit():
            if int(code_token) > 10000:
                # vulnerability_knowledge_4
                csg_vul = [(root_node, 0, 'check', ['vul'], [])]
        return csg_vul, states, new_tokens
    elif root_node.type in pointer_relation:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        # vulnerability_knowledge_5
        csg_vul.append((root_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in null_pointer:
        csg_vul = []
        # vulnerability_knowledge_5
        csg_vul.append((root_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in int_type:
        csg_vul = []
        # vulnerability_knowledge_7
        csg_vul.append((root_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in array_relation:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        # vulnerability_knowledge_14, vulnerability_knowledge_15
        csg_vul.append((root_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens

    elif root_node.type in def_statement:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        identifier = root_node.children[1]
        if identifier.type == 'identifier':
            # vulnerability_knowledge_3
            csg_vul.append((root_node, 0, 'check', ['vul'], []))
            identifier_index = CSG_var_to_code_tokens_pos(identifier, index_to_code)
            identifier_name = generate_index_to_code_token_dict(identifier_index[0], code)
            if identifier_name in ["passwd", "password"]:
                # vulnerability_knowledge_19, vulnerability_knowledge_20
                csg_vul.append((identifier, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens

    elif root_node.type in for_statement:
        csg_vul = []
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp

        for child in root_node.children:
            if child.type == 'binary_expression':
                # vulnerability_knowledge_8
                csg_vul.append((child, 0, 'check', ['vul'], []))
                break
        return csg_vul, states, new_tokens
    elif root_node.type in while_statement:
        csg_vul = []
        condition_node = root_node.children[1]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp

        # vulnerability_knowledge_8
        csg_vul.append((condition_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in do_first_statement:
        csg_vul = []
        condition_node = root_node.children[3]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp

        # vulnerability_knowledge_8
        csg_vul.append((condition_node, 0, 'check', ['vul'], []))
        return csg_vul, states, new_tokens
    elif root_node.type in api_relation:
        function_node = root_node.children[0]
        function_index = CSG_var_to_code_tokens_pos(function_node, index_to_code)
        function_name = generate_index_to_code_token_dict(function_index[0], code)
        input_func_list = ["gets", "scanf", "fscanf", "getchar", "fgetc", "fgets"]
        mem_func_list = ["memcpy", "strcpy", "strncpy"]
        printf_func_list = ["printf", "fprintf"]
        path_travel = ["open","fopen"]
        command_inject = ["system", "popen", "eval"]
        race_condition = ["synchronize", "lock", "unlock"]
        authorization = ["setcookie", "grant"]
        privilege = ["setuid", "setgid", "seteuid"]
        if function_name in input_func_list:
            csg_vul = []
            argument_list = root_node.children[1]
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp

            # vulnerability_knowledge_9
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in mem_func_list:
            csg_vul = []
            argument_list = root_node.children[1]
            dst = argument_list.children[1]
            src = argument_list.children[3]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_1
            csg_vul.append((src, 0, 'check', ['vul'], []))
            csg_vul.append((dst, 1, '<', [src], []))
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in printf_func_list:
            csg_vul = []
            argument_list = root_node.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_10
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in path_travel:
            csg_vul = []
            argument_list = root_node.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_11
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in command_inject:
            csg_vul = []
            argument_list = root_node.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_12, vulnerability_knowledge_13
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in race_condition:
            csg_vul = []
            argument_list = root_node.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_17
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in authorization:
            csg_vul = []
            argument_list = root_node.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_18
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name in privilege:
            csg_vul = []
            argument_list = root_node.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_18
            csg_vul.append((argument_list, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name == 'malloc':
            csg_vul = []
            argument_list = root_node.children[1]
            mem = argument_list.children[1]

            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp

            # vulnerability_knowledge_8, vulnerability_knowledge_16
            csg_vul.append((mem, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        elif function_name == "free":
            if function_name in states:
                csg_vul = []
                for child in root_node.children:
                    temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
                    csg_vul += temp
                # vulnerability_knowledge_6
                csg_vul.append((root_node, 0, 'check', ['vul'], []))
                return csg_vul, states, new_tokens
            else:
                idx, code = index_to_code[(function_node.start_point, function_node.end_point)]
                states[function_name] = [idx]

                csg_vul = []
                for child in root_node.children:
                    temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
                    csg_vul += temp

                # vulnerability_knowledge_2
                csg_vul.append((root_node, 0, 'check', ['vul'], []))
                return csg_vul, states, new_tokens
        elif function_name in states:
            csg_vul = []
            argument_list = root_node.children[1]
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(argument_list, index_to_code, states, code, new_tokens)
            csg_vul += temp
            # vulnerability_knowledge_3
            csg_vul.append((root_node, 0, 'check', ['vul'], []))
            return csg_vul, states, new_tokens
        else:
            csg_vul = []
            for child in root_node.children:
                temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
                csg_vul += temp
            return csg_vul, states, new_tokens
    else:
        csg_vul=[]
        for child in root_node.children:
            temp, states, new_tokens = generate_CSG_vul_subgraph_c(child, index_to_code, states, code, new_tokens)
            csg_vul += temp
        return csg_vul, states, new_tokens
