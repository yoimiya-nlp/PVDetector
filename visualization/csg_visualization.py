from CSG_c import generate_CSG_df_subgraph_c, generate_CSG_cf_subgraph_c, generate_CSG_vul_subgraph_c
from utils import (remove_comments, CSG_node_to_code_tokens_pos,
                   generate_index_to_code_token_dict, CSG_var_to_code_tokens_pos)
from tree_sitter import Language, Parser
from transformers import RobertaTokenizer
import networkx as nx
import matplotlib.pyplot as plt

c_csg_function = {
    'df': generate_CSG_df_subgraph_c,
    'cf': generate_CSG_cf_subgraph_c,
    'vul': generate_CSG_vul_subgraph_c
}

different_language_parser = {
    'c': c_csg_function
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
    # df_subgraph = filtered_dataflows 
    # print("filtered_dataflows: ", filtered_dataflows)
    df_subgraph = merge_df_equals(filtered_dataflows)
    #print("df_subgraph: ", df_subgraph)

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


def convert_function_to_CSG(tokenizer, function):
    func = function
    this_parser = parsers['c']
    code_length = 512
    data_flow_length = 96
    control_flow_length = 20
    vul_relation_length = 12

    try:
        code_tokens, csg_df_subgraph, csg_cf_subgraph, csg_vul_subgraph = \
            generate_csg_graph(func, this_parser, 'c')
        # print("code_tokens: ", code_tokens)
        print("csg_df_subgraph: ", csg_df_subgraph)
        # print("csg_cf_subgraph: ", csg_cf_subgraph)
        # print("csg_vul_subgraph: ", csg_vul_subgraph)
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

    code_tokens = code_tokens[:code_length][:512 - 3]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    csg_df_subgraph = csg_df_subgraph[:code_length + data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in csg_df_subgraph]
    position_idx += [0 for _ in csg_df_subgraph]
    source_ids += [tokenizer.unk_token_id for _ in csg_df_subgraph]

    csg_cf_subgraph = csg_cf_subgraph[
                      :code_length + data_flow_length + control_flow_length - len(source_tokens)]
    source_tokens += [x[0].type for x in csg_cf_subgraph]
    position_idx += [0 for _ in csg_cf_subgraph]
    source_ids += [tokenizer.unk_token_id for _ in csg_cf_subgraph]

    csg_vul_subgraph = csg_vul_subgraph[
                       :code_length + data_flow_length + control_flow_length + vul_relation_length - len(
                           source_tokens)]
    source_tokens += [x[0].type for x in csg_vul_subgraph]
    position_idx += [0 for _ in csg_vul_subgraph]
    source_ids += [tokenizer.unk_token_id for _ in csg_vul_subgraph]

    padding_length = code_length + data_flow_length + control_flow_length + vul_relation_length - len(
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
                       csg_cf2code_pos, csg_cf_prenode_id, csg_vul2code_pos, csg_vul_prenode_id, 1, 0)
                       

def process_functions(function):
    tokenizer = RobertaTokenizer.from_pretrained("model_config")
    csg = convert_function_to_CSG(tokenizer, function)
    print("csg: ", csg)


c_lib_funcs_keywords = ['printf', 'scanf', 'if', 'else', 'for', 'while', 'do',
                        'switch', 'case', 'default', 'break', 'continue', 'return',
                        'sizeof', 'NULL', 'malloc', 'free', 'strcpy', 'strncpy', 'strcat',
                        'strncat', 'memcmp', 'strcmp', 'strncmp', 'memcpy', 'memmove',
                        'memset', 'strchr', 'strrchr', 'strstr', 'strlen', 'strnlen',
                        'strspn', 'strcspn', 'strpbrk', 'strsep', 'strtok', 'strerror',
                        'atoi', 'atol', 'atoll', 'itoa', 'strtol', 'strtoll', 'strtoul',
                        'strtoull', 'strtof', 'strtod', 'strtold',
                        'char', 'int', 'float', 'double', 'long', 'short', 'void', 'unsigned',
                        'signed', 'const', 'static']


def span_label(code_tokens, idxs, prefix, max_len=13):
    """
    use code token slice to generate visualization label. idxs is the token index list.
    keep label short via max_len.
    """
    if not idxs:
        return f"{prefix}[]"
    i0, i1 = min(idxs), max(idxs)
    text = " ".join(code_tokens[i0:i1+1])
    if len(text) > max_len:
        text = text[:max_len-3] + '...'
    return f"{prefix}[{i0}-{i1}] {text}"


def visualization(code=None, lang='c',
                  show=('df', 'cf', 'vul'),
                  cf_relations=('determined_by', 'declaration'),      # only related to conditional statements
                  vul_relations=('<',),                 # only related to memcpy/strcpy generated '<' relation
                  label_max_len=13,
                  layout='spring',                      # 'spring' or 'kamada'
                  seed=42,
                  subplot_mode=False):  # New parameter for subplot mode
    # 1) get code and generate CSG
    if code is None:
        code = """
int save_input()
{
    int TEMP_NUM = 40;
    int mem_num = 80;
    char temp[TEMP_NUM];
    mem_num = mem_num - 50;
    mem = (char *)malloc(mem_num);
    gets(temp);
    if (strlen(temp) >= TEMP_NUM) {
        printf("Error!input too long!");
    }
    else {
        strncpy(mem, temp, 10);
     }
     return 0;
}
""".strip("\n")

    this_parser = parsers[lang]
    code_tokens, df_subgraph, cf_subgraph, vul_subgraph = generate_csg_graph(code, this_parser, lang)

    # 2) build graph (assign different weights to different relations, for spring_layout to be more cohesive)
    G = nx.DiGraph()

    def add_edge(u, v, kind, label, w):
        G.add_edge(u, v, kind=kind, label=label, w=w)

    # Helper function to get identifier value from node
    def get_identifier_value(node, index_to_code_token):
        """Extract identifier value from AST node"""
        try:
            # Get token positions for the node
            token_positions = CSG_var_to_code_tokens_pos(node, index_to_code_token)
            if token_positions:
                # Get the first token position and extract the value
                pos = token_positions[0]
                value = generate_index_to_code_token_dict(pos, code.split('\n'))
                return value
        except:
            pass
        return f"{node.type}@{node.start_point}"

    # Create index_to_code_token mapping for identifier extraction
    tokens_index = CSG_node_to_code_tokens_pos(this_parser[0].parse(bytes(code, 'utf8')).root_node)
    code_lines = code.split('\n')
    code_tokens_for_mapping = [generate_index_to_code_token_dict(x, code_lines) for x in tokens_index]
    index_to_code_token = {}
    for idx, (index, token) in enumerate(zip(tokens_index, code_tokens_for_mapping)):
        index_to_code_token[index] = (idx, token)

    # ---- data flow (blue) ----
    df_edges = []
    if 'df' in show:
        for (child_token, child_id, relation, parent_tokens, parent_ids) in df_subgraph:
            if relation == 'declaration':
                G.add_node(f"{child_token}_{child_id}")
                continue
            child_node = f"{child_token}_{child_id}"
            for ptok, pid in zip(parent_tokens, parent_ids):
                parent_node = f"{ptok}_{pid}"
                add_edge(parent_node, child_node, kind='df', label='df', w=1.0)
                df_edges.append((parent_node, child_node))

    # ---- control flow (orange, dashed) ----
    def node_label_from_ast(prefix, n, offset=None):
        sp = getattr(n, 'start_point', ('?', '?'))
        node_type = getattr(n, 'type', '?')
        # Shorten long node types
        if len(node_type) > 8:
            node_type = node_type[:5] + '...'
        
        if offset is not None:
            return f"{prefix}:{node_type}[{offset}]@{sp}"
        return f"{prefix}:{node_type}@{sp}"

    cf_edges = []
    cf_node_mapping = {}  # Map (node_type, offset) to node label

    if 'cf' in show:
        # First pass: create all nodes
        for (node_obj, offset, relation, fathers, child_token_idxs) in cf_subgraph:
            if relation not in cf_relations:
                continue
            
            # Create node label with offset for parenthesized_expression nodes
            if node_obj.type == 'parenthesized_expression':
                node_label = node_label_from_ast('cf', node_obj, offset)
            else:
                node_label = node_label_from_ast('cf', node_obj)
            
            # Store mapping for edge creation
            cf_node_mapping[(node_obj.type, offset)] = node_label
            G.add_node(node_label)

        # Second pass: create edges based on relation type
        for i, (node_obj, offset, relation, fathers, child_token_idxs) in enumerate(cf_subgraph):
            if relation not in cf_relations:
                continue
            
            # Get current node label
            if node_obj.type == 'parenthesized_expression':
                current_label = node_label_from_ast('cf', node_obj, offset)
            else:
                current_label = node_label_from_ast('cf', node_obj)
            
            if relation == 'declaration':
                # Declaration nodes are head nodes (only out-degree, no in-degree)
                # No edges to create from this node
                continue
                
            elif relation == 'determined_by':
                # This is an edge: current node is child, find parent using offset
                if offset > 0 and i >= offset:
                    parent_idx = i - offset
                    if parent_idx < len(cf_subgraph):
                        parent_node_obj, parent_offset, parent_relation, parent_fathers, parent_child_token_idxs = cf_subgraph[parent_idx]
                        if parent_relation in cf_relations:
                            # Get parent node label
                            if parent_node_obj.type == 'parenthesized_expression':
                                parent_label = node_label_from_ast('cf', parent_node_obj, parent_offset)
                            else:
                                parent_label = node_label_from_ast('cf', parent_node_obj)
                            
                            # Create edge from parent to child
                            add_edge(parent_label, current_label, kind='cf', label='cf', w=0.3)
                            cf_edges.append((parent_label, current_label))

    # ---- vul (red) ----
    vul_edges = []
    vul_relation_nodes = {}  # Map relation to intermediate node

    if 'vul' in show:
        for (node_obj, offset, relation, fathers, child_token_idxs) in vul_subgraph:
            if relation not in vul_relations:
                continue
            
            # Create intermediate relation node
            relation_node = f"vul:{relation}[{offset}]"
            if relation_node not in vul_relation_nodes:
                vul_relation_nodes[relation_node] = relation_node
                G.add_node(relation_node)
            
            # Get identifier value for current node
            current_identifier = get_identifier_value(node_obj, index_to_code_token)
            current_node = f"vul:{current_identifier}"
            G.add_node(current_node)
            
            # Connect current identifier to relation node
            add_edge(current_node, relation_node, kind='vul', label=f'vul:{relation}', w=0.3)
            vul_edges.append((current_node, relation_node))
            
            # Connect relation node to father identifiers
            for f in fathers:
                father_identifier = get_identifier_value(f, index_to_code_token)
                father_node = f"vul:{father_identifier}"
                G.add_node(father_node)
                
                add_edge(relation_node, father_node, kind='vul', label=f'vul:{relation}', w=0.3)
                vul_edges.append((relation_node, father_node))

    # 3) Choose visualization mode
    if subplot_mode and len(show) > 1:
        # Multi-subplot mode
        fig, axes = plt.subplots(1, len(show), figsize=(5*len(show), 6))
        if len(show) == 1:
            axes = [axes]
        
        for idx, relation_type in enumerate(show):
            ax = axes[idx]
            
            # Create subgraph for this relation type
            subG = nx.DiGraph()
            
            if relation_type == 'df':
                # Add df nodes and edges
                for (child_token, child_id, relation, parent_tokens, parent_ids) in df_subgraph:
                    if relation == 'declaration':
                        subG.add_node(f"{child_token}_{child_id}")
                    else:
                        child_node = f"{child_token}_{child_id}"
                        for ptok, pid in zip(parent_tokens, parent_ids):
                            parent_node = f"{ptok}_{pid}"
                            subG.add_edge(parent_node, child_node)
            
            elif relation_type == 'cf':
                # Add cf nodes and edges
                for (node_obj, offset, relation, fathers, child_token_idxs) in cf_subgraph:
                    if relation not in cf_relations:
                        continue
                    
                    if node_obj.type == 'parenthesized_expression':
                        node_label = node_label_from_ast('cf', node_obj, offset)
                    else:
                        node_label = node_label_from_ast('cf', node_obj)
                    subG.add_node(node_label)
                
                # Add cf edges
                for i, (node_obj, offset, relation, fathers, child_token_idxs) in enumerate(cf_subgraph):
                    if relation not in cf_relations:
                        continue
                    
                    if node_obj.type == 'parenthesized_expression':
                        current_label = node_label_from_ast('cf', node_obj, offset)
                    else:
                        current_label = node_label_from_ast('cf', node_obj)
                    
                    if relation == 'determined_by' and offset > 0 and i >= offset:
                        parent_idx = i - offset
                        if parent_idx < len(cf_subgraph):
                            parent_node_obj, parent_offset, parent_relation, parent_fathers, parent_child_token_idxs = cf_subgraph[parent_idx]
                            if parent_relation in cf_relations:
                                if parent_node_obj.type == 'parenthesized_expression':
                                    parent_label = node_label_from_ast('cf', parent_node_obj, parent_offset)
                                else:
                                    parent_label = node_label_from_ast('cf', parent_node_obj)
                                subG.add_edge(parent_label, current_label)
            
            elif relation_type == 'vul':
                # Add vul nodes and edges
                for (node_obj, offset, relation, fathers, child_token_idxs) in vul_subgraph:
                    if relation not in vul_relations:
                        continue
                    
                    relation_node = f"vul:{relation}[{offset}]"
                    subG.add_node(relation_node)
                    
                    current_identifier = get_identifier_value(node_obj, index_to_code_token)
                    current_node = f"vul:{current_identifier}"
                    subG.add_node(current_node)
                    subG.add_edge(current_node, relation_node)
                    
                    for f in fathers:
                        father_identifier = get_identifier_value(f, index_to_code_token)
                        father_node = f"vul:{father_identifier}"
                        subG.add_node(father_node)
                        subG.add_edge(relation_node, father_node)
            
            # Layout for subgraph
            if subG.number_of_nodes() > 0:
                pos = nx.spring_layout(subG, k=1.5, iterations=200, seed=seed)
                
                # Draw subgraph
                nx.draw(subG, pos, with_labels=True, node_size=1500, node_color='lightblue',
                       font_size=8, font_weight='bold', ax=ax)
                
                # Draw edges with different colors
                if relation_type == 'df':
                    nx.draw_networkx_edges(subG, pos, edge_color='tab:blue', width=2, ax=ax)
                elif relation_type == 'cf':
                    nx.draw_networkx_edges(subG, pos, edge_color='tab:orange', width=2,
                                         style='dashed', alpha=0.9, ax=ax)
                elif relation_type == 'vul':
                    nx.draw_networkx_edges(subG, pos, edge_color='tab:red', width=2,
                                         alpha=0.9, ax=ax)
            
            ax.set_title(f"{relation_type.upper()} Subgraph")
            ax.set_axis_off()
        
        plt.tight_layout()
        plt.savefig('csg_visualization.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        # Single graph mode with improved layout
        def compute_layout(graph, which='spring', seed_val=42):
            n = graph.number_of_nodes()
            if n == 0:
                return {}
            if which == 'kamada':
                return nx.kamada_kawai_layout(graph)
            # Improved spring layout parameters
            k = 2.0 / (n ** 0.5)  # Increased k for better spacing
            return nx.spring_layout(graph, k=k, iterations=500, seed=seed_val, weight='w')
        
        pos = compute_layout(G, which=layout, seed_val=seed)
        edge_labels = nx.get_edge_attributes(G, 'label')

        fig, ax = plt.subplots(figsize=(16, 10))
        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.03)

        node_size = 2000 
        nx.draw(G, pos, with_labels=True, node_size=node_size, node_color='lightblue',
                font_size=9, font_weight='bold', ax=ax)

        if df_edges:
            nx.draw_networkx_edges(G, pos, edgelist=df_edges, edge_color='tab:blue', width=2, ax=ax)
        if cf_edges:
            nx.draw_networkx_edges(G, pos, edgelist=cf_edges, edge_color='tab:orange', width=2,
                                   style='dashed', alpha=0.9, ax=ax)
        if vul_edges:
            nx.draw_networkx_edges(G, pos, edgelist=vul_edges, edge_color='tab:red', width=2,
                                   alpha=0.9, ax=ax)

        title_parts = []
        if 'df' in show: title_parts.append('df')
        if 'cf' in show: title_parts.append('cf')
        if 'vul' in show: title_parts.append('vul')
        ax.set_title(f"CSG visualization ({'/'.join(title_parts)})")
        ax.set_axis_off()
        plt.show()


if __name__ == '__main__':
    sample = """
int save_input()
{
    int TEMP_NUM = 40;
    int mem_num = 80;
    char temp[TEMP_NUM];
    mem_num = mem_num - 50;
    mem = (char *)malloc(mem_num);
    gets(temp);
    if (strlen(temp) >= TEMP_NUM) {
        printf("Error!input too long!");
    }
    else {
        strncpy(mem, temp, 10);
     }
     return 0;
}
"""
    # process_functions(sample)
    visualization(show=('df', 'cf', 'vul'), subplot_mode=True)
