import re
from io import StringIO
import  tokenize
from tree_sitter import Language, Parser


def remove_comments(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        #source = remove_indentation(source)
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def CSG_node_to_code_tokens_pos(node):
    """
    Input: 
        node: CSG node
    Output:
        a list of token positions
    """
    if (len(node.children)==0 or node.type.find('string') != -1) and node.type!='comment':
        return [(node.start_point,node.end_point)]
    else:
        code_tokens = []
        for child in node.children:
            code_tokens+=CSG_node_to_code_tokens_pos(child)
        return code_tokens


def CSG_var_to_code_tokens_pos(node, index_2_code_token):
    """
    Input: 
        node: CSG node
        index_2_code_token: a dictionary, key is token position, value is token string
    Output:
        a list of token positions
    """
    # ignore string_literal and string_content
    if (len(node.children)==0 or node.type.find('string') != -1) and node.type!='comment':
        index = (node.start_point,node.end_point)
        _, code = index_2_code_token[index]
        if node.type!=code:
            return [(node.start_point,node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in node.children:
            code_tokens+=CSG_var_to_code_tokens_pos(child, index_2_code_token)
        return code_tokens    


def generate_index_to_code_token_dict(pos, all_code):
    """
    Input: 
        pos: token position, a tuple, e.g. ((3, 4), (3, 7))
        all_code: full code, a list, each element is a line of code
    Output:
        token string
    """
    start_point = pos[0]
    end_point = pos[1]

    if start_point[0] == end_point[0]:                  # same line
        s = all_code[start_point[0]][start_point[1]:end_point[1]]
    else:                                               # not same line
        s = ""
        s += all_code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1, end_point[0]):
            s += all_code[i]
        s += all_code[end_point[0]][:end_point[1]]
    return s

if __name__ == '__main__':
    # declaration
    C_LANGUAGE = Language('../build/my-languages.so', 'c')
    c_parser = Parser()
    c_parser.set_language(C_LANGUAGE)

    # C example
    cpp_code_snippet = '''
    int save_input()
{
    int TEMP_NUM = 40;
    int mem_num = 80;
    char temp[TEMP_NUM];
    mem_num = mem_num - 50;
    mem = (char *)malloc(sizeof(char)*mem_num);
    gets(temp);
    if (strlen(temp) < TEMP_NUM) {
        strcpy(mem, temp);
    }
    else {
        printf("Error!input too long!");
    }
    return 0;
}
    '''

    tree = c_parser.parse(bytes(cpp_code_snippet, "utf8"))
    root_node = tree.root_node

    tokens_index = CSG_node_to_code_tokens_pos(root_node)
    print("root_node index: ", tokens_index)

    code = cpp_code_snippet.split('\n')
    code_tokens = [generate_index_to_code_token_dict(x, code) for x in tokens_index]

    index_to_code_token = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code_token[index] = (idx, code)
    print("index_to_code: ", index_to_code_token)

    query_node = tree.root_node.children[0].children[2].children[6].children[0].children[0]
    print("query_node: ", query_node)
    query_index = CSG_var_to_code_tokens_pos(query_node, index_to_code_token)
    print("query_node index: ", query_index)
    query_node_token = generate_index_to_code_token_dict(query_index[0], cpp_code_snippet.split('\n'))
    print("query_node_token: ", query_node_token)
    input_func_list = ["gets", "scanf", "fscanf", "getchar", "fgetc", "fgets"]
    if query_node_token in input_func_list:
        print("yes")
    declaration_node = root_node.children[0].children[2].children[1]
    print("declaration_node: ", declaration_node)
    declaration_node2 = root_node.children[0].children[2].children[1]
    print("declaration_node2: ", declaration_node2)
    print("isequal: ", declaration_node==declaration_node2)
    declaration_index1 = CSG_var_to_code_tokens_pos(declaration_node, index_to_code_token)

    print("declaration_node index: ", declaration_index1)
    print("name: ", declaration_node.child_by_field_name('right'))
    print("name: ", declaration_node.children[2])
    # get code line
    cpp_loc = cpp_code_snippet.split('\n')
    code_tokens = [generate_index_to_code_token_dict(x, cpp_loc) for x in tokens_index]
    # ['int', 'main', '{', 'printf', '(', '"hello world"', ')', ';', 'return', 'O', ';', '}']
    print(code_tokens)
    for i in range(len(declaration_index1)):
        print("declaration_node_tokens: ", generate_index_to_code_token_dict(declaration_index1[i], cpp_loc))
