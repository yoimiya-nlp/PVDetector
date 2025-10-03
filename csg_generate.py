from parser import generate_CSG_df_subgraph_c, generate_CSG_cf_subgraph_c, generate_CSG_vul_subgraph_c
from parser import generate_CSG_df_subgraph_cpp, generate_CSG_cf_subgraph_cpp, generate_CSG_vul_subgraph_cpp
from parser import generate_CSG_df_subgraph_java, generate_CSG_cf_subgraph_java, generate_CSG_vul_subgraph_java
from parser import generate_CSG_df_subgraph_py, generate_CSG_cf_subgraph_py, generate_CSG_vul_subgraph_py
from parser import generate_CSG_df_subgraph_js, generate_CSG_cf_subgraph_js, generate_CSG_vul_subgraph_js
from parser import generate_CSG_df_subgraph_php, generate_CSG_cf_subgraph_php, generate_CSG_vul_subgraph_php
from parser import (remove_comments,
                    CSG_node_to_code_tokens_pos,
                    generate_index_to_code_token_dict,
                    CSG_var_to_code_tokens_pos)
from tree_sitter import Language, Parser

c_csg_function = {
    'df': generate_CSG_df_subgraph_c,
    'cf': generate_CSG_cf_subgraph_c,
    'vul': generate_CSG_vul_subgraph_c
}

cpp_csg_function = {
    'df': generate_CSG_df_subgraph_cpp,
    'cf': generate_CSG_cf_subgraph_cpp,
    'vul': generate_CSG_vul_subgraph_cpp
}

java_csg_function = {
    'df': generate_CSG_df_subgraph_java,
    'cf': generate_CSG_cf_subgraph_java,
    'vul': generate_CSG_vul_subgraph_java
}

python_csg_function = {
    'df': generate_CSG_df_subgraph_py,
    'cf': generate_CSG_cf_subgraph_py,
    'vul': generate_CSG_vul_subgraph_py
}

js_csg_function = {
    'df': generate_CSG_df_subgraph_js,
    'cf': generate_CSG_cf_subgraph_js,
    'vul': generate_CSG_vul_subgraph_js
}

php_csg_function = {
    'df': generate_CSG_df_subgraph_php,
    'cf': generate_CSG_cf_subgraph_php,
    'vul': generate_CSG_vul_subgraph_php
}

different_language_parser = {
    'c': c_csg_function,
    'cpp': cpp_csg_function,
    'java': java_csg_function,
    'python': python_csg_function,
    'javascript': js_csg_function,
    'php': php_csg_function
}

# load parsers
parsers = {}
for language in different_language_parser:
    LANGUAGE = Language('build/my-languages.so', language)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, different_language_parser[language]]
    parsers[language] = parser


class CSGFeatures(object):
    """A single CSG features for an example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 position_idx,
                 csg_df2code_pos,
                 csg_df_prenode_id,
                 csg_cf2code_pos,
                 csg_cf_prenode_id,
                 csg_vul2code_pos,
                 csg_vul_prenode_id,
                 label,
                 url
                 ):
        # input_tokens: cls, code_tokens, sep, CSGnodes(dfsub, cfsub, vulsub)
        # input_ids: cls0, code_tokens from 4 to n, sep2, unk3(CSGnodes), pad 1
        # position_idx: cls, code_tokens, sep from 2 to n, csg node 0, pad 1

        # csg_df2code_pos: [(12, 13), (12, 13)]
        # csg_df_prenode_id: [[3], [4], [5], [], [], [], [2]]

        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.csg_df2code_pos = csg_df2code_pos
        self.csg_df_prenode_id = csg_df_prenode_id

        self.csg_cf2code_pos = csg_cf2code_pos
        self.csg_cf_prenode_id = csg_cf_prenode_id

        self.csg_vul2code_pos = csg_vul2code_pos
        self.csg_vul_prenode_id = csg_vul_prenode_id

        # label
        self.label = label
        self.url = url


def merge_df_equals(df_edges):
    """
    input: CSG df edges list [(token, child_id, relation, [parent_tokens], [parent_ids]), ...]
    process: merge equals_to to comes_from (only contains comes_from/declaration), and compress the equivalence class to the representative id
    output: normalized and deduplicated df edges list (only contains declaration/comes_from)
    """
    # union-find by id
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1) use equals_to to build union find set: child_id is equivalent to the id it comes from (parent_ids[0])
    for tok, cid, rel, ptoks, pids in df_edges:
        if rel == 'equals_to' and pids:
            union(pids[0], cid)

    # 2) rewrite other edges, map to representative id; discard equals_to; and deduplicate
    #    use (token, child_id) -> {parent_id: parent_token} to aggregate parents
    agg = {}
    for tok, cid, rel, ptoks, pids in df_edges:
        if rel == 'equals_to':
            continue
        rep_c = find(cid)
        if rel == 'declaration':
            key = (tok, rep_c)
            entry = agg.setdefault(key, {})
            # declaration has no parent; keep as placeholder, avoid being completely discarded
            entry.setdefault(None, None)
        else:
            key = (tok, rep_c)
            entry = agg.setdefault(key, {})
            for ptok, pid in zip(ptoks, pids):
                rep_p = find(pid)
                if rep_p == rep_c:
                    continue  # avoid self-loop
                # deduplicate by id; if same id but different token, keep the last token string
                entry[rep_p] = ptok

    # 3) expand to standard list, only includes declaration/comes_from, sorted by child_id
    result = []
    for (tok, rep_c), parent_map in agg.items():
        if list(parent_map.keys()) == [None]:
            # declaration
            result.append((tok, rep_c, 'declaration', [], []))
        else:
            ids_sorted = sorted(parent_map.keys())
            ptoks_sorted = [parent_map[i] for i in ids_sorted]
            result.append((tok, rep_c, 'comes_from', ptoks_sorted, ids_sorted))
    result.sort(key=lambda x: x[1])
    return result
    

def generate_csg_graph(code, this_parser, lang):
    # generate data flow subgraph, control flow subgraph, and vul relation subgraph
    # remove comments
    code = remove_comments(code, lang)
    parser_df = this_parser[1]['df']
    parser_cf = this_parser[1]['cf']
    parser_vul = this_parser[1]['vul']

    tree = this_parser[0].parse(bytes(code, 'utf8'))  # tree-sitter generate AST
    root_node = tree.root_node
    tokens_index = CSG_node_to_code_tokens_pos(root_node)  # generate [(start,end)] sequence

    code = code.split('\n')

    code_tokens = [generate_index_to_code_token_dict(x, code) for x in tokens_index]

    index_to_code_token = {}
    for idx, (index, token) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code_token[index] = (idx, token) 

    # generate data flow subgraph
    new_tokens = []
    try:
        dataflows, _, new_tokens = parser_df(root_node, index_to_code_token, {}, code, new_tokens)
    except:
        print("dataflows generate error")
        dataflows = []

    dataflows = sorted(dataflows, key=lambda x: x[1])  # sort by child id
    df_indexs = set()
    for d in dataflows:
        if len(d[-1]) != 0:
            df_indexs.add(d[1])
        for x in d[-1]:
            df_indexs.add(x)
    filtered_dataflows = []
    for d in dataflows:
        if d[1] in df_indexs:
            filtered_dataflows.append(d)
    df_subgraph = merge_df_equals(filtered_dataflows)

    # generate control flow subgraph
    try:
        controlflows = parser_cf(root_node)
    except:
        print("controlflows generate error")
        controlflows = []

    cf_subgraph = []
    for cf in controlflows:
        cf_index = CSG_node_to_code_tokens_pos(cf[0])
        cf_tokens_idx = []
        for index in cf_index:
            idx, _ = index_to_code_token[index]
            cf_tokens_idx.append(idx)
        cf_list =  list(cf)
        cf_list[-1] = cf_tokens_idx
        cf_subgraph.append(cf_list)

    # generate vul relation subgraph
    vul_relation = []
    vul_relation.append(('vul', len(index_to_code_token) + len(new_tokens), 'declaration', [], []))
    new_tokens.append('vul')
    try:
        vulrelations, _, new_tokens = parser_vul(root_node, index_to_code_token, {}, code, new_tokens)
    except:
        print("vulrelations generate error")
        vulrelations = []

    for i in range(len(vulrelations)):
        vul_relation.append(vulrelations[i])
    vul_subgraph = []

    for vul in vul_relation:
        if vul[0] == 'vul':
            continue
        vul_index = CSG_node_to_code_tokens_pos(vul[0])
        vul_tokens_idx = []
        for index in vul_index:
            idx, _ = index_to_code_token[index]
            vul_tokens_idx.append(idx)
        vul_list = list(vul)
        vul_list[-1] = vul_tokens_idx
        vul_subgraph.append(vul_list)

    for i in range(len(new_tokens)):
        code_tokens.append(new_tokens[i])
    return code_tokens, df_subgraph, cf_subgraph, vul_subgraph


def convert_function_to_CSG(item):
    url, label, tokenizer, args, url_to_code = item
    func = url_to_code[url]
    this_parser = parsers[args.program_language]

    try:
        code_tokens, csg_df_subgraph, csg_cf_subgraph, csg_vul_subgraph = \
            generate_csg_graph(func, this_parser, args.program_language)
    except:
        print("CSG generate error")
        print("func: ", func)
        code_tokens = []
        csg_df_subgraph = []
        csg_cf_subgraph = []
        csg_vul_subgraph = []
    # print("code_tokens: ", code_tokens)

    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]                  # subword model
    token_subword_new_pos = {}                              # real position of (i)th token
    token_subword_new_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        token_subword_new_pos[i] = (token_subword_new_pos[i - 1][1], token_subword_new_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]       # flatten subword tokens

    code_tokens = code_tokens[:args.code_length][:512 - 3]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    csg_df_subgraph = csg_df_subgraph[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in csg_df_subgraph]
    position_idx += [0 for _ in csg_df_subgraph]
    source_ids += [tokenizer.unk_token_id for _ in csg_df_subgraph]

    csg_cf_subgraph = csg_cf_subgraph[
                      :args.code_length + args.data_flow_length + args.control_flow_length - len(source_tokens)]
    source_tokens += [x[0].type for x in csg_cf_subgraph]
    position_idx += [0 for _ in csg_cf_subgraph]
    source_ids += [tokenizer.unk_token_id for _ in csg_cf_subgraph]

    csg_vul_subgraph = csg_vul_subgraph[
                       :args.code_length + args.data_flow_length + args.control_flow_length + args.vul_relation_length - len(
                           source_tokens)]
    source_tokens += [x[0].type for x in csg_vul_subgraph]
    position_idx += [0 for _ in csg_vul_subgraph]
    source_ids += [tokenizer.unk_token_id for _ in csg_vul_subgraph]

    padding_length = args.code_length + args.data_flow_length + args.control_flow_length + args.vul_relation_length - len(
        source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    # re-encode forward node id (parent id) to new id (child id's index, i.e., child id's position in csg_df_subgraph)
    csg_df_nodes = {}  
    for idx, x in enumerate(csg_df_subgraph):
        csg_df_nodes[x[1]] = idx
    for idx, x in enumerate(csg_df_subgraph):
        csg_df_subgraph[idx] = x[:-1] + ([csg_df_nodes[i] for i in x[-1] if i in csg_df_nodes],)

    csg_df_prenode_id = [x[-1] for x in csg_df_subgraph]
    csg_df2code_pos = [token_subword_new_pos[x[1]] for x in csg_df_subgraph]
    cls_len = len([tokenizer.cls_token])
    csg_df2code_pos = [(x[0] + cls_len, x[1] + cls_len) for x in csg_df2code_pos]

    # csg_cf_prenode_id is the offset value, and the current position plus the offset value is the forward node's position
    csg_cf_prenode_id = [-x[1] for x in csg_cf_subgraph]

    csg_cf2code_pos = [(token_subword_new_pos[x[-1][0]], token_subword_new_pos[x[-1][-1]]) for x in csg_cf_subgraph]
    csg_cf2code_pos = [(x[0][0] + cls_len, x[1][1] + cls_len) for x in csg_cf2code_pos]

    # same as csg_cf_prenode_id
    csg_vul_prenode_id = [-x[1] for x in csg_vul_subgraph]
    csg_vul2code_pos = [(token_subword_new_pos[x[-1][0]], token_subword_new_pos[x[-1][-1]]) for x in csg_vul_subgraph]
    csg_vul2code_pos = [(x[0][0] + cls_len, x[1][1] + cls_len) for x in csg_vul2code_pos]

    return CSGFeatures(source_tokens, source_ids, position_idx, csg_df2code_pos, csg_df_prenode_id,
                       csg_cf2code_pos, csg_cf_prenode_id, csg_vul2code_pos, csg_vul_prenode_id, label, url)
