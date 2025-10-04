import difflib
import re
from apyori import apriori
import os


C_LIB_FUNCS_KEYWORDS = [
    'printf', 'scanf', 'if', 'else', 'for', 'while', 'do',
    'switch', 'case', 'default', 'break', 'continue', 'return',
    'sizeof', 'NULL', 'malloc', 'free', 'strcpy', 'strncpy', 'strcat',
    'strncat', 'memcmp', 'strcmp', 'strncmp', 'memcpy', 'memmove',
    'memset', 'strchr', 'strrchr', 'strstr', 'strlen', 'strnlen',
    'strspn', 'strcspn', 'strpbrk', 'strsep', 'strtok', 'strerror',
    'atoi', 'atol', 'atoll', 'itoa', 'strtol', 'strtoll', 'strtoul',
    'strtoull', 'strtof', 'strtod', 'strtold'
]


def generate_diff(vulnerability_folder, patch_folder, diff_folder):
    """Compare two files and output the diff to a file."""
    vuln_files = os.listdir(vulnerability_folder)
    patch_files = os.listdir(patch_folder)

    # read vul and patch
    for vuln_file, patchn_file in zip(vuln_files, patch_files):
        vulnerability_file = os.path.join(vulnerability_folder, vuln_file)
        patch_file = os.path.join(patch_folder, patchn_file)
        diff_path = os.path.join(diff_folder, vuln_file.replace('.c', '.diff'))
        with open(vulnerability_file, 'r', encoding='utf-8') as file1, \
                open(patch_file, 'r', encoding='utf-8') as file2:
            vuln_lines = file1.readlines()
            patch_lines = file2.readlines()

        # use difflib to generate diff
        diff = difflib.unified_diff(vuln_lines, patch_lines,
                                    fromfile=vuln_file,
                                    tofile=patchn_file)

        # write to the file
        with open(diff_path, 'w') as diff_file:
            diff_file.writelines(diff)


def load_diff(diff_folder):
    """Load a code diff file and return the diff as a list of lines."""
    diff_files = os.listdir(diff_folder)
    diffs = []

    for diff_file in diff_files:
        diff_path = os.path.join(diff_folder, diff_file)
        with open(diff_path, 'r') as file:
            diff = file.readlines()
            diff.append('end_of_function')
        diffs.extend(diff)

    print(f'Loaded {len(diffs)} diff lines')
    return diffs


def symbolize_code(diff_lines):
    """Replace variable names and function names with generic symbols."""
    symbolized_lines = []
    var_count = 0
    func_count = 0
    var_map = {}
    func_map = {}

    # List of C library functions and keywords
    c_lib_funcs_keywords = ['printf', 'scanf', 'if', 'else', 'for', 'while', 'do',
                            'switch', 'case', 'default', 'break', 'continue', 'return',
                            'sizeof', 'NULL', 'malloc', 'free', 'strcpy', 'strncpy', 'strcat',
                            'strncat', 'memcmp', 'strcmp', 'strncmp', 'memcpy', 'memmove',
                            'memset', 'strchr', 'strrchr', 'strstr', 'strlen', 'strnlen',
                            'strspn', 'strcspn', 'strpbrk', 'strsep', 'strtok', 'strerror',
                            'atoi', 'atol', 'atoll', 'itoa', 'strtol', 'strtoll', 'strtoul',
                            'strtoull', 'strtof', 'strtod', 'strtold']

    function_lines = []
    for line in diff_lines:
        if line.strip() == 'end_of_function':
            symbolized_lines.append(function_lines)
            function_lines = []
            continue
        # Replace function names with generic symbols (e.g., FUNC1, FUNC2)
        functions = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b(?=\()', line)
        for func in functions:
            if func not in func_map and func not in c_lib_funcs_keywords:
                func_count += 1
                func_map[func] = f'FUNC{func_count}'
            if func in func_map:
                line = line.replace(func, func_map[func])

        # Replace variable names with generic symbols (e.g., VAR1, VAR2)
        variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
        for var in variables:
            if var not in var_map and var not in c_lib_funcs_keywords and var not in func_map:
                var_count += 1
                var_map[var] = f'VAR'
            if var in var_map:
                line = line.replace(var, var_map[var])

        function_lines.append(line.strip())

    return symbolized_lines


def extract_call_transactions(diff_lines, c_lib_funcs_keywords):
    transactions = []
    current_tx = []

    def flush_tx():
        nonlocal current_tx
        if current_tx:
            transactions.append(current_tx)
            current_tx = []

    skip_prefixes = ('diff ', 'index ', '---', '+++', '@@', '***')

    for line in diff_lines:
        s = line.strip()
        if s == 'end_of_function':
            flush_tx()
            continue

        if any(s.startswith(prefix) for prefix in skip_prefixes):
            continue

        if not line or line[0] not in ['+', '-']:
            continue

        code = line[1:].strip()

        # extract function call: name(args)
        for m in re.finditer(r'([a-zA-Z_]\w*)\s*\(([^()]*)\)', code):
            func = m.group(1)
            args = m.group(2)

            norm_func = func if func in c_lib_funcs_keywords else 'FUNC'

            # standardize each argument
            raw_args = [a.strip() for a in args.split(',')] if args.strip() else []
            var_map = {}
            var_idx = 1
            norm_args = []

            for a in raw_args:
                if not a:
                    continue
                # literal -> CONST
                if re.match(r'^\d+(\.\d+)?$', a) or re.match(r"^'.*'$", a) or re.match(r'^".*"$', a):
                    norm_args.append('CONST')
                    continue

                # remove unary symbol and pointer symbol
                a_clean = re.sub(r'^[&*\s]+', '', a)
                a_clean = re.sub(r'\s+', '', a_clean)

                # take the first identifier as the representative of the argument
                idents = re.findall(r'[A-Za-z_]\w*', a_clean)
                if not idents:
                    norm_args.append('EXPR')
                else:
                    key = idents[0]
                    if key not in var_map:
                        var_map[key] = f'VAR{var_idx}'
                        var_idx += 1
                    norm_args.append(var_map[key])

            pattern = f"{norm_func}({', '.join(norm_args)})"
            current_tx.append(pattern)

    flush_tx()
    return transactions


def find_frequent_patterns(transactions, min_support, c_lib_funcs_keywords):
    results = list(apriori(transactions=transactions, min_support=min_support))

    patterns = []
    for r in results:
        # only 1 item set can get `strcpy(VAR1, VAR2)` such pattern
        if len(r.items) == 1:
            item = next(iter(r.items))
            # ensure the pattern is like name(VAR..., ...) and name is in C library list
            m = re.match(r'^([A-Za-z_]\w*)\((.*)\)$', item)
            if not m:
                continue
            fname = m.group(1)
            if fname in c_lib_funcs_keywords:
                patterns.append((item, r.support))

    patterns.sort(key=lambda x: x[1], reverse=True)
    return patterns


def generate_formulation_for_sentence(sentence):
    # generate vulnerability knowledge from a single sentence
    formulations = []
    c_variable_keywords = ['int', 'char', 'float', 'double', 'long', 'short', 'signed', 'unsigned']

    # check if the sentence contains variable definition
    if any(keyword in sentence for keyword in c_variable_keywords):
        # find all variables in the sentence
        each_var = [word for word in sentence if 'VAR' in word]
        # generate formulation for variable definition
        for var in each_var:
            formulations.append(f'NULL -> {var}')

    # check if the sentence contains assignment
    if '=' in sentence:
        assign_index = sentence.index('=')
        left_var = sentence[assign_index - 1]
        right_var = ' '.join(sentence[assign_index + 1:])
        formulations.append(f'{right_var} -> {left_var}')

    # check if the sentence contains function calls
    if '(' in sentence:
        func_index = sentence.index('(')
        func_name = sentence[func_index - 1]
        if func_name in ['strcpy', 'strncpy', 'strcat', 'strncat', 'memcpy', 'memmove', 'memset']:
            formulations.append(f'{sentence[func_index + 1]} -> < -> {sentence[func_index + 2]}')
        elif func_name in ['free']:
            formulations.append(f'NULL -> {sentence[func_index + 1]}')

    return formulations


def generate_formulation_for_pattern(pattern):
    # generate vulnerability knowledge from frequent patterns
    formulations = []
    for sentence in pattern:
        formulation = generate_formulation_for_sentence(sentence)
        if formulation:
            formulations.append(formulation)
    common_vars = set.intersection(*[set(sentence) for sentence in pattern])
    # print("common_vars: ", common_vars)
    if len(pattern) == 1:
        return formulations

    # multiple sentences
    for common_var in common_vars:
        times = 0
        for sentence in pattern:
            if common_var in sentence:
                if times == 0:
                    formulations.append([' '.join(sentence), '->', common_var])
                    times += 1
                else:
                    formulations.append([common_var, '->', ' '.join(sentence)])

    return formulations


def generate_vulnerability_knowledge(patterns):
    # generate vulnerability knowledge
    my_knowledge_base = []
    for pattern in patterns:
        formulations = generate_formulation_for_pattern(pattern)
        my_knowledge_base.append(formulations)
    return my_knowledge_base


def vulnerability_knowledge_process(min_support=0.3):
    number = 119  # CWE number
    vulnerability_folder = 'example/CWE' + str(number) + '/vul'
    patch_folder = 'example/CWE' + str(number) + '/sec'
    diff_folder = 'example/CWE' + str(number) + '/dif'
    generate_diff(vulnerability_folder, patch_folder, diff_folder)

    # Step 1: Load diff
    diff = load_diff(diff_folder)
    # Step 2: Symbolize the code
    # symbolized_diff = symbolize_code(diff[:100])
    # print("symbolized_diff: ", symbolized_diff)
    transactions = extract_call_transactions(diff, C_LIB_FUNCS_KEYWORDS)
    # Step 3: Apply Apriori to find frequent patterns
    frequent_patterns = find_frequent_patterns(transactions, min_support=min_support, c_lib_funcs_keywords=C_LIB_FUNCS_KEYWORDS)
    print("frequent_patterns: ", frequent_patterns[:5])
    # Step 4: Generate vulnerability knowledge
    frequent_patterns_base = [[['strcpy', '(', 'VAR1', 'VAR2', ')']],
                              [['free', '(', 'VAR1', ')'], ['VAR1', '=', 'VAR2']],
                              [['int', 'VAR1'], ['VAR2', '=', 'VAR1', '+', '1']]]
    my_knowledge_base = generate_vulnerability_knowledge(frequent_patterns_base)
    # print("knowledge_base: ", my_knowledge_base)

    return my_knowledge_base


if __name__ == '__main__':
    knowledge_base = vulnerability_knowledge_process()
    for entry in knowledge_base:
        print("knowledge entry: ", entry)
