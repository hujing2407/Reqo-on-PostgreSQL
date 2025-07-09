import copy
import os
import re
import string
import sys
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import postgresql_workload_generator

def load_database_info(dbname):
    file_path = f'Data/{dbname}/database_statistics/'
    tables_index = np.load(file_path + "tables_index.npy", allow_pickle=True).item()
    tables_index_all = np.load(file_path + "tables_index_all.npy", allow_pickle=True).item()
    columns_index = np.load(file_path + "columns_index.npy", allow_pickle=True).item()
    columns_list = np.load(file_path + "columns_list.npy", allow_pickle=True)
    attribute_range = np.load(file_path + "attribute_range.npy", allow_pickle=True).item()
    nodes = np.load(file_path + "postgresql_nodestypes_all.npy", allow_pickle=True).item()
    return tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes

def replace_aliases_and_columns(original_plan, columns_list):
    # Alias to full table name mapping
    alias_map = {}

    # First pass: collect all aliases and corresponding full table names
    def collect_aliases(node):
        if "Relation Name" in node and "Alias" in node:
            alias_map[node["Alias"]] = node["Relation Name"]

        # Recursively collect aliases in sub-plans
        if "Plans" in node:
            for subplan in node["Plans"]:
                collect_aliases(subplan)

    # Second pass: update strings in the plan using the alias mapping
    def apply_aliases(node, table_name=None):
        # Use deep copy to avoid modifying the original node
        new_node = copy.deepcopy(node)

        # Update the table name for the current node, if available
        current_table = new_node.get("Relation Name", table_name)

        # Update string fields
        for key, value in new_node.items():
            if isinstance(value, str):
                original_value = value  # Save the original value for change logging
                for alias, full_name in alias_map.items():
                    # Ensure only replacing "alias." forms
                    pattern = re.compile(r'\b' + re.escape(alias) + r'\.')
                    if pattern.search(value):
                        value = pattern.sub(full_name + ".", value)
                        new_node[key] = value
                        # print(f"Changed '{key}': '{original_value}' to '{value}'")

        # Recursively update sub-plans
        if "Plans" in new_node:
            new_node["Plans"] = [apply_aliases(subplan, current_table) for subplan in new_node["Plans"]]

        # Apply column name formatting
        format_column_names(new_node, current_table)

        return new_node

    def format_column_names(node, table_name):
        if table_name:
            for key, value in node.items():
                if isinstance(value, str):
                    # Function to conditionally replace column names
                    def replace_columns(match):
                        column_name = match.group(0)
                        # Regex to ensure column is not part of a qualified name
                        # Check both: not preceded and not followed by '.' or any word character
                        full_pattern = re.compile(r'(?<![\w.])' + re.escape(column_name) + r'(?![\w.])')
                        if full_pattern.search(value):
                            # We need to further verify it's not part of a longer identifier
                            before = value[:match.start()]
                            after = value[match.end():]
                            if not (before.endswith('.') or re.match(r'\.\s*\w+', after)):
                                return f"{table_name}.{column_name}"
                        return column_name

                    # Create a regex pattern from the unique columns list to match whole words
                    columns_pattern = r'\b(' + '|'.join(re.escape(column) for column in columns_list) + r')\b'
                    # Replace standalone column names with "table_name.column"
                    new_value = re.sub(columns_pattern, replace_columns, value)
                    if new_value != value:
                        node[key] = new_value
                        # print(f"Formatted '{key}': '{original_value}' to '{new_value}'")

    # Start collecting aliases
    collect_aliases(original_plan)
    # print(alias_map)

    # Update the plan using the collected aliases
    new_plan = apply_aliases(original_plan)
    return new_plan

def replace_aliases_and_columns_in_query_paln(data, columns_list):
    data_replaced = []
    for query_plan in data:
        data_replaced.append(replace_aliases_and_columns(query_plan, columns_list))
    return replace_aliases_and_columns(data, columns_list)

def extract_predicates(text):
    predicate_patterns = [
        r'(\w+\.\w+)\s*([=<>]{1,2}|<>|~~|!~~|in|like|not like)\s*(\{.*?\}|\[.*?\]|".*?"|\'[^\']*?\'|\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?|\S+)'
    ]
    predicates = []
    segments = text.split('|')
    for segment in segments:
        for pattern in predicate_patterns:
            matches = re.findall(pattern, segment)
            for match in matches:
                if len(match) == 3:
                    table_column, operator, raw_value = match
                    value = clean_value(raw_value)
                    predicates.append([table_column, operator, value])
                else:
                    print("Unexpected match format:", match)

    return predicates

def clean_value(value):
    value = value.strip("',\")")
    if value.startswith('{') and value.endswith('}'):
        return '{' + re.sub(r"['\"]", "", value[1:-1]) + '}'
    value = re.sub(r'::.*', '', value)
    return value

def is_float_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def str_value_encoding(values, model_path):
    model = Word2Vec.load(model_path)
    translation_table = str.maketrans('', '', string.punctuation)
    processed_values = [value.lower().translate(translation_table).split() for value in values]
    words = [word for sublist in processed_values for word in sublist]
    try:
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            encoded_value = np.mean(word_vectors, axis=0)[0]
        else:
            encoded_value = -2
    except Exception as e:
        print(f'Word2vec model {model_path} error or empty input!', e)
        encoded_value = -2
    return encoded_value

def Text_extraction(text, tables_index, tables_index_all, columns_index, attribute_range, node_type):
    if node_type != 'Bitmap Index Scan':
        enc_table = [0]*len(tables_index)
        words = re.findall(r'\b\w+\b', text)
        for item in words:
            if item in tables_index_all:
                enc_table[tables_index_all[item]] = 1
        return list(np.where(np.array(enc_table) == 1)[0])
    else:
        enc_table = []
        words = re.findall(r'\b\w+\b', text)
        for item in words:
            if item in tables_index_all:
                enc_table.append(tables_index_all[item])
        return [enc_table[0]]

def Node_info_extract(node, tables_index, tables_index_all, columns_index, attribute_range, nodes):
    nodetype_enc = nodes[node['Node Type']]

    info = ""
    if node['Node Type'] == "Seq Scan":
        if "Filter" in node:
            info += node["Filter"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]

    elif node['Node Type'] == "Hash":
        pass

    elif node['Node Type'] == "Hash Join":
        if "Hash Cond" in node:
            info += str(node["Hash Cond"])
        if "Join Filter" in node:
            info += "|" + str(node["Join Filter"])

    elif node['Node Type'] == "Sort":
        if "Sort Key" in node:
            info += str(node["Sort Key"])
        else:
            pass

    elif node['Node Type'] == "Aggregate":
        if "Group Key" in node:
            info += str(node["Group Key"])
        else:
            pass

    elif node['Node Type'] == "Gather Merge":
        pass

    elif node['Node Type'] == "CTE Scan":
        if "CTE Name" in node:
            info += node["CTE Name"]
        if "Filter" in node:
            info += "|" + node["Filter"]

    elif node['Node Type'] == "Nested Loop":
        if "Join Filter" in node:
            info += str(node["Join Filter"])
        else:
            pass

    elif node['Node Type'] == "Index Scan":
        if "Index Name" in node:
            info += "|" + node["Index Name"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]

    elif node['Node Type'] == "Limit":
        pass

    elif node['Node Type'] == "Append":
        pass

    elif node['Node Type'] == "Merge Join":
        if "Merge Cond" in node:
            info += node["Merge Cond"]
        if "Join Filter" in node:
            info += "|" + node["Join Filter"]

    elif node['Node Type'] == "Bitmap Index Scan":
        if "Index Name" in node:
            info += "|" + node["Index Name"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]
        if "Index Cond" in node:
            info += "|" + node["Index Cond"]

    elif node['Node Type'] == "Bitmap Heap Scan":
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]
        if "Alias" in node:
            info += "|" + node["Alias"]
        if "Recheck Cond" in node:
            info += "|" + node["Recheck Cond"]

    elif node['Node Type'] == "Unique":
        pass

    elif node['Node Type'] == "Gather":
        pass

    elif node['Node Type'] == "Materialize":
        pass

    elif node['Node Type'] == "Subquery Scan":
        pass

    elif node['Node Type'] == "SetOp":
        pass

    elif node['Node Type'] == "WindowAgg":
        pass

    elif node['Node Type'] == "Memoize":
        if "Cache Key" in node:
            info += str(node["Cache Key"])
        else:
            pass

    elif node['Node Type'] == "Index Only Scan":
        if "Index Name" in node:
            info += node["Index Name"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]

    elif node['Node Type'] == "Incremental Sort":
        if "Sort Key" in node:
            info += str(node["Sort Key"])
        if "Presorted Key" in node:
            info += "|" + str(node["Presorted Key"])

    elif node['Node Type'] == "Group":
        if "Group Key" in node:
            info += str(node["Group Key"])
        else:
            pass

    elif node['Node Type'] == "Result":
        pass

    elif node['Node Type'] == "Merge Append":
        if "Sort Key" in node:
            info += str(node["Sort Key"])
        else:
            pass

    elif node['Node Type'] == "HashAggregate":
        if "Group Key" in node:
            info += str(node["Group Key"])
        else:
            pass

    elif node['Node Type'] == "BitmapAnd":
        pass

    else:
        print('Unknown Node Type:', node['Node Type'])

    filter_enc = Text_extraction(info, tables_index, tables_index_all, columns_index, attribute_range, node['Node Type'])

    return nodetype_enc, filter_enc

def Subtree_traversal(tree, L, index):
    node_index = index
    if 'Plans' in tree:
        for i in range(len(tree['Plans'])):
            L, index = Subtree_traversal(tree['Plans'][i], L, index+1)
    if 'Plans' in tree and len(tree['Plans']) >= 2:
        L.append(tree)
        if node_index == 0:
            return L
    # If treat a leaf node as a subtree.
    elif 'Plans' not in tree:
        L.append(tree)
        if node_index == 0:
            return L
    else:
        if node_index == 0:
            if L != []:
                L[-1] = tree
                return L
            else:
                return [tree]
    return L, index

def Data_augmentation(data):
    data_plus = []
    for tree in data:
        L = Subtree_traversal(tree, [], 0)
        if L != []:
            data_plus = data_plus + L
    return data_plus

def get_plan_stats(data):
    costs = []
    rows = []

    def recurse(n):
        costs.append(n["Total Cost"])
        rows.append(n["Plan Rows"])
        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)

    for plan in data:
        recurse(plan)

    costs = np.array(costs)
    rows = np.array(rows)

    costs = np.log(costs + 1)
    rows = np.log(rows + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)

    return [["Total Cost", "Plan Rows"], [costs_min, rows_min], [costs_max, rows_max]]

def Join_only_tree(tree, join_tree):
    if 'Plans' in tree:
        child_num = len(tree['Plans'])
        if child_num == 1:
            if tree['Node Type'] in ["Hash Join", "Nested Loop", "Merge Join"]:
                join_tree = Join_only_tree(tree['Plans'][0], join_tree)
                if join_tree != {}:
                    join_tree = {'joins':[join_tree], 'label':tree["Actual Total Time"] * 1}
                else:
                    join_tree = {'label':tree["Actual Total Time"] * 1}
            else:
                join_tree = Join_only_tree(tree['Plans'][0], join_tree)
        else:
            join_tree_child = []
            for i in range(child_num):
                join_tree_child_i = Join_only_tree(tree['Plans'][i], join_tree)
                if join_tree_child_i != {}:
                    join_tree_child.append(join_tree_child_i)
            if join_tree_child != []:
                join_tree = {'joins': join_tree_child, 'label': tree["Actual Total Time"] * 1}
            else:
                join_tree = {'label': tree["Actual Total Time"] * 1}
    # If treat a leaf node as a subtree.
    else:
        join_tree = {'label': tree["Actual Total Time"] * 1}
    return join_tree

def Join_p_c_pair(join_tree, index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label):
    node_index = index
    if 'joins' in join_tree:
        if len(join_tree['joins']) == 1:
            L_pair_index.append([node_index, index+1])
            if (join_tree['label']-join_tree['joins'][0]['label']) < 0:
                L_pair_label.append(abs(join_tree['label']-join_tree['joins'][0]['label']))
                # print('error! join_tree label is smaller than its child!')
                # print(str(node_index) + ' ' + str(join_tree))
            else:
                L_pair_label.append(join_tree['label']-join_tree['joins'][0]['label'])
            index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label = Join_p_c_pair(join_tree['joins'][0], index+1, post_t_index, post_t_index_dic, L_pair_index, L_pair_label)
        else:
            multi_child_node_index = []
            for i in range(len(join_tree['joins'])):
                multi_child_node_index.append(index+1)
                index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label = Join_p_c_pair(join_tree['joins'][i], index+1, post_t_index, post_t_index_dic, L_pair_index, L_pair_label)
            L_pair_index.append([node_index, multi_child_node_index])
            if (join_tree['label']-sum([join_tree['joins'][i]['label'] for i in range(len(join_tree['joins']))])) < 0:
                L_pair_label.append(abs(join_tree['label']-sum([join_tree['joins'][i]['label'] for i in range(len(join_tree['joins']))])))
                # print('error! join_tree label is smaller than its multi child!')
                # print(str(node_index) + ' ' + str(join_tree))
            else:
                L_pair_label.append(join_tree['label']-sum([join_tree['joins'][i]['label'] for i in range(len(join_tree['joins']))]))
    else:
        L_pair_index.append([node_index, node_index])
        L_pair_label.append(join_tree['label'])
    post_t_index_dic[node_index] = post_t_index
    post_t_index += 1

    if node_index == 0:
        L_pair_index_post = []
        for i in L_pair_index:
            if type(i[1]) == int:
                L_pair_index_post.append([post_t_index_dic[i[0]], post_t_index_dic[i[1]]])
            else:
                L_pair_index_post.append([post_t_index_dic[i[0]], [post_t_index_dic[j] for j in i[1]]])
        return L_pair_index_post, L_pair_label
    return index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label

def join_tree_correct(tree, L_label):
    if 'joins' in tree:
        if len(tree['joins']) == 1:
            subtree, L_label = join_tree_correct(tree['joins'][0], L_label)
            if tree['label'] < subtree['label']:
                if tree['label'] < tree['joins'][0]['label']:
                    tree['label'] = subtree['label'] + tree['label']
                else:
                    tree['label'] = subtree['label'] + tree['label'] - tree['joins'][0]['label']
            tree['joins'][0] = subtree
        else:
            subtree = []
            for i in range(len(tree['joins'])):
                subtree_i, L_label = join_tree_correct(tree['joins'][i], L_label)
                subtree.append(subtree_i)
            if tree['label'] < sum([subtree[i]['label'] for i in range(len(subtree))]):
                if tree['label'] < sum([tree['joins'][i]['label'] for i in range(len(tree['joins']))]):
                    tree['label'] = sum([subtree[i]['label'] for i in range(len(subtree))]) + tree['label']
                else:
                    tree['label'] = sum([subtree[i]['label'] for i in range(len(subtree))]) + tree['label'] - sum([tree['joins'][i]['label'] for i in range(len(tree['joins']))])
            for i in range(len(tree['joins'])):
                tree['joins'][i] = subtree[i]
    L_label.append(tree['label'])
    return tree, L_label

def get_join_tree_label(tree, L_label):
    if 'joins' in tree:
        if len(tree['joins']) == 1:
            L_label = join_tree_correct(tree['joins'][0], L_label)
        else:
            for i in range(len(tree['joins'])):
                L_label = join_tree_correct(tree['joins'][i], L_label)
    L_label.append(tree['label'])
    return L_label

def join_explanation_generate(tree):
    join_tree = Join_only_tree(tree, {})
    if join_tree == {}:
        join_tree = {'label': tree["Actual Total Time"] * 1}
    join_tree['label'] = tree["Actual Total Time"] * 1
    join_tree, L_label = join_tree_correct(join_tree, [])
    L_pair_index, L_pair_label = Join_p_c_pair(join_tree, 0, 0, {}, [], [])
    return L_pair_index, L_pair_label, L_label

def filter_SPPHints(hints, max_hints_num, max_relations, mode):
    """
    Filter the list of pg_hint_plan hint strings based on:
      1. The maximum number of relations allowed.
         (Count the relations by removing any extra parentheses and splitting on whitespace.)
      2. The mode: "all" (default), "only_join", or "only_scan".
         For "only_join" only include hints whose operation name is a join hint.
         For "only_scan" only include hints whose operation name is a scan hint.

    Finally, return the selected hints as a multi-line string in the pg_hint format:
        /*+
        Hint1
        Hint2
        ...
        */
    """
    # Define which hint names are considered join or scan hints.
    join_hint_set = {"HashJoin", "MergeJoin", "NestLoop", "NoHashJoin", "NoMergeJoin", "NoNestLoop"}
    scan_hint_set = {"SeqScan", "IndexScan", "BitmapIndexScan", "BitmapHeapScan", "IndexOnlyScan",
                     "NoSeqScan", "NoIndexScan", "NoBitmapIndexScan", "NoBitmapHeapScan", "Sort", "NoSort"}

    filtered = []
    for hint in hints:
        # The expected format is: OpHint(content)
        # Extract the operation name (the part before the first "(").
        idx = hint.find("(")
        if idx == -1:
            continue  # skip if format is not as expected
        op_hint = hint[:idx]

        # Extract the content inside the outer parentheses.
        end_idx = hint.rfind(")")
        if end_idx == -1:
            continue
        content = hint[idx + 1:end_idx]
        # Remove any inner parentheses and then split by whitespace.
        content_clean = content.replace("(", "").replace(")", "")
        tokens = content_clean.split()
        relation_count = len(tokens)
        if relation_count > max_relations:
            continue  # skip hint if too many relations

        # Mode filtering.
        if mode == "only_join" and op_hint not in join_hint_set:
            continue
        if mode == "only_scan" and op_hint not in scan_hint_set:
            continue

        filtered.append(hint)

    if not filtered:
        return "", 0
    # Limit the number of hints to max_hints_num.
    filtered = filtered[:max_hints_num]


    # Format the output as a multi-line string with proper indentation.
    hints_str = "\n".join(filtered)

    return "/*+\n" + hints_str + "\n*/", len(filtered)


def convert_key_to_string(key, tables_index_inv):
    """
    Convert a key (a tuple, which may itself contain tuples) by reversing the order
    in each inner tuple and mapping numbers back to table names.
    """
    # Case 1: key is a flat tuple of ints.
    if all(isinstance(x, int) for x in key):
        names = [tables_index_inv.get(num, str(num)) for num in reversed(key)]
        if len(names) == 1:
            return names[0]
        else:
            return "(" + " ".join(names) + ")"
    # Case 2: key is a tuple of tuples.
    elif all(isinstance(x, tuple) for x in key):
        parts = []
        for tup in key:
            names = [tables_index_inv.get(num, str(num)) for num in reversed(tup)]
            # If the inner tuple has only one element, no need for extra parentheses.
            if len(names) == 1:
                parts.append(names[0])
            else:
                parts.append("(" + " ".join(names) + ")")
        # Simply join the parts with a space (no extra outer parentheses).
        return " ".join(parts)
    # Case 3: Mixed types.
    else:
        parts = []
        for x in key:
            if isinstance(x, tuple):
                names = [tables_index_inv.get(num, str(num)) for num in reversed(x)]
                if len(names) == 1:
                    parts.append(names[0])
                else:
                    parts.append("(" + " ".join(names) + ")")
            else:
                parts.append(tables_index_inv.get(x, str(x)))
        return " ".join(parts)


def generate_SPPHints(analyzed_collect_operations, nodes_inv, tables_index_inv):
    """
    For each item in analyzed_collect_operations:
      - Convert the key by reversing each inner tuple and mapping numbers back to table names.
      - For the value [flag, op_num, avg, count, total_sum]:
          * Look up the operation name using nodes_inv.
          * Use the flag to choose either the 'use' or 'avoid' hint from hint_mapping.
      - Construct a pg_hint_plan hint string of the form: <hint>(<converted_key>)
    Returns a list of generated hint strings.
    """
    hint_mapping = {
        'Seq Scan': {'1': 'SeqScan', '0': 'NoSeqScan'},
        'Index Scan': {'1': 'IndexScan', '0': 'NoIndexScan'},
        'Bitmap Index Scan': {'1': 'BitmapIndexScan', '0': 'NoBitmapIndexScan'},
        'Bitmap Heap Scan': {'1': 'BitmapHeapScan', '0': 'NoBitmapHeapScan'},
        'Index Only Scan': {'1': 'IndexOnlyScan', '0': 'NoIndexOnlyScan'},
        'Hash': {'1': 'Hash', '0': 'NoHash'},
        'Hash Join': {'1': 'HashJoin', '0': 'NoHashJoin'},
        'Merge Join': {'1': 'MergeJoin', '0': 'NoMergeJoin'},
        'Nested Loop': {'1': 'NestLoop', '0': 'NoNestLoop'},
        'Sort': {'1': 'Sort', '0': 'NoSort'},
        # Add additional mappings as needed.
    }

    hints_list = []
    for key, value in analyzed_collect_operations.items():
        # Convert key to string using the updated function.
        key_str = convert_key_to_string(key, tables_index_inv)
        try:
            flag, op_num, avg, count, total_sum, improved_ratio = value
        except Exception as e:
            print(f"Skipping key {key} due to invalid value format: {value}")
            continue

        op_name = nodes_inv.get(op_num, None)
        if op_name is None:
            print(f"No operation found for op_num {op_num} in key {key}")
            continue
        if op_name not in hint_mapping:
            print(f"No matching hint mapping for operation {op_name}")
            continue

        # Use flag '1' for use and '0' for avoid.
        op_hint = hint_mapping[op_name]['1'] if flag == 1 else hint_mapping[op_name]['0']

        # Construct the hint string in pg_hint_plan style.
        # hint_str = f"/*+ {op_hint}({key_str}) */"
        hint_str = f"{op_hint}({key_str})"
        hints_list.append(hint_str)

    return hints_list


def analyze_collect_operations(new_collect_operations, f_min, f_max, cnt_min, ranking):
    """
    Analyze new_collect_operations and produce a new dictionary with improved avg, count, and total_sum.

    For each item (key with a list of sublists [num1, avg, count, total_sum, improved_ratio]):
      - If the list length == 1, skip the item.
      - Otherwise, let:
            C_total = sum(count for all sublists)
            T_total = sum(total_sum for all sublists)
      - Let highest_sub be the first sublist (with highest avg).

      Condition 1:
        If there is at least one candidate sublist among the remaining ones (with avg lower than highest_sub)
        and whose count is >= C_total * f_min and >= cnt_min, then choose the candidate with the lowest avg.

        Then compute:
          new_total_sum  = T_total - ( candidate.total_sum + candidate.avg * (C_total - candidate.count) )
          new_avg        = new_total_sum / C_total
          new_count      = sum(count for each sublist in S with avg > candidate.avg)
          improved_ratio = new_total_sum / T_total

        If new_total_sum > 0, set the item's value to: [1, candidate.num1, new_avg, new_count/C_total, new_total_sum, improved_ratio]

      Condition 2:
        Else, if highest_sub.count < C_total * f_max, then let:
          Let R_total_sum   = sum(total_sum for all remaining sublists)
              R_total_count = sum(count for all remaining sublists)
          new_total_sum  = T_total - ((R_total_sum / R_total_count) * C_total)  (if R_total_count > 0, else T_total)
          new_avg        = new_total_sum / C_total
          new_count      = highest_sub.count
          improved_ratio = new_total_sum / T_total

        If new_total_sum > 0, set the item's value to: [0, highest_sub.num1, new_avg, new_count, new_total_sum, improved_ratio]

      Otherwise (Condition 3), skip the item.

    Finally, return the resulting dictionary sorted (ranked) by the ranking (fifth element by default).
    """
    analyzed_dict = {}
    for key, value_list in new_collect_operations.items():
        # Skip items with only one sublist.
        if len(value_list) <= 1:
            continue

        # Calculate total count and total sum across all sublists.
        C_total = sum(sub[2] for sub in value_list)
        T_total = sum(sub[3] for sub in value_list)

        # highest_sub is the first element (with highest avg, since sorted descending by avg)
        highest_sub = value_list[0]

        # Condition 1: Look for candidate sublists among the rest with avg < highest_sub.avg,
        # and with count >= C_total * f_min and >= cnt_min.
        candidates = [sub for sub in value_list[1:] if
                      sub[1] < highest_sub[1] and sub[2] >= C_total * f_min and sub[2] >= cnt_min]

        if candidates:
            # Choose the candidate with the lowest avg among candidates.
            candidate = min(candidates, key=lambda x: x[1])
            # Compute improved total_sum.
            new_total_sum = T_total - (candidate[3] + candidate[1] * (C_total - candidate[2]))
            # Skip item if new_total_sum is not positive.
            if new_total_sum <= 0:
                continue
            new_avg = new_total_sum / C_total
            # new_count is the sum of counts for sublists with avg greater than candidate's.
            new_count = sum(sub[2] for sub in value_list if sub[1] > candidate[1])
            improved_ratio = new_total_sum / T_total
            analyzed_dict[key] = [1, candidate[0], new_avg, new_count / C_total, new_total_sum, improved_ratio]
        else:
            # Condition 2: if highest_sub.count is larger than C_total * f_max and >= cnt_min.
            if highest_sub[2] <= C_total * f_max and highest_sub[2] >= cnt_min:
                remaining = value_list[1:]
                R_total_count = sum(sub[2] for sub in remaining)
                R_total_sum = sum(sub[3] for sub in remaining)
                if R_total_count > 0:
                    new_total_sum = T_total - ((R_total_sum / R_total_count) * C_total)
                else:
                    new_total_sum = T_total
                # Skip item if new_total_sum is not positive.
                if new_total_sum <= 0:
                    continue
                new_avg = new_total_sum / C_total
                new_count = highest_sub[2]
                improved_ratio = new_total_sum / T_total
                analyzed_dict[key] = [0, highest_sub[0], new_avg, new_count / C_total, new_total_sum, improved_ratio]
            else:
                # Condition 3: do not include this key.
                continue

    # Rank (sort) the resulting dictionary by the value at index `ranking` (default new_total_sum) in descending order.
    sorted_items = sorted(analyzed_dict.items(), key=lambda item: item[1][ranking], reverse=True)
    return dict(sorted_items)

def update_collect_operations(collect_operations):
    new_dict = {}
    for key, value_list in collect_operations.items():
        # Group by num1 and accumulate sum and count for num2
        groups = defaultdict(lambda: [0, 0])  # groups[num1] = [total_sum, count]
        for num1, num2 in value_list:
            groups[num1][0] += num2  # add to total sum
            groups[num1][1] += 1  # increment count

        # Build the new value list: each sublist is [num1, avg, count, total_sum]
        new_value = []
        for num1 in groups:
            total_sum, count = groups[num1]
            avg = total_sum / count
            new_value.append([num1, avg, count, total_sum])

        # Rank (sort) the sublists based on the average (second element)
        new_value.sort(key=lambda x: x[1], reverse=True)
        new_dict[key] = new_value
    return new_dict

def collect_operation_and_relations(subtrees, L_expl, L_pair_index, collect_operations, tables_index, tables_index_all, columns_index, attribute_range, nodes):
    assert len(subtrees) == len(L_pair_index)
    unexecuted_operations = []
    used_tables = [[] for _ in range(len(subtrees))]
    for subtree_i in range(len(subtrees)):
        subtree = subtrees[subtree_i]
        left = L_pair_index[subtree_i][0]
        right = L_pair_index[subtree_i][1]
        # print(L_pair_index[subtree_i])
        # print(subtree)
        if isinstance(right, list):  # nodes with more than one child
            if left in unexecuted_operations or any([element in unexecuted_operations for element in right]):
                unexecuted_operations.append(subtree_i)
                continue
            while 'Join' not in subtree['Node Type'] and "Nested Loop" not in subtree['Node Type']:
                if 'Plans' in subtree:
                    subtree = subtree['Plans'][0]
                else:
                    print(f"Error: Join node not found in subtree:{subtree}, left:{left}, right:{right}, node:{subtree['Node Type']}")
                    sys.exit()
                    break
            operation, _ = Node_info_extract(subtree, tables_index, tables_index_all, columns_index, attribute_range, nodes)
            relation = []
            child_cost = 0
            for child_i in right:
                relation.append(used_tables[child_i])
                child_cost += L_expl[child_i]
            relation.sort(key=lambda sublist: (len(sublist), sublist))
            used_tables[subtree_i] = [item for sublist in relation for item in sublist]
            relation = tuple(tuple(sublist) for sublist in relation)
            # print(relation)
            if relation not in collect_operations:
                collect_operations[relation] = [[operation, abs(L_expl[subtree_i] - child_cost)]]
            else:
                collect_operations[relation].append([operation, abs(L_expl[subtree_i] - child_cost)])

        else:
            if left == right:  # leaf nodes
                if 'Scan' in subtree['Node Type'] and left not in unexecuted_operations:
                    operation, relation = Node_info_extract(subtree, tables_index, tables_index_all, columns_index, attribute_range, nodes)
                    if L_expl[subtree_i] <= 0:
                        unexecuted_operations.append(subtree_i)
                        continue
                    assert relation != []
                    # print(relation)
                    if len(relation) != 1:
                        print('Error: Scan leaf node has more than one relations in subtree:{subtree}, left:{left}, right:{right}, node:{subtree["Node Type"]}')
                        sys.exit()
                    used_tables[subtree_i] = relation
                    relation = tuple(relation)
                    if tuple(relation) not in collect_operations:
                        collect_operations[relation] = [[operation, L_expl[subtree_i]]]
                    else:
                        collect_operations[relation].append([operation, L_expl[subtree_i]])

                elif left in unexecuted_operations:
                    continue
                else:
                    print(f"Error: Scan not in leaf node in subtree:{subtree}, left:{left}, right:{right}, node:{subtree['Node Type']}")
                    break

            else:
                print(f"Error: Parent node only has one child in subtree:{subtree}, left:{left}, right:{right}, node:{subtree['Node Type']}")
                sys.exit()
        # print(collect_operations)
    return collect_operations

def executa_query_with_SPPHints(queries, SPPHints, db_params):
    """
    Execute a query with the provided PostgreSQL hints.
    """
    executed_query_plans_with_SPPHints = []
    executed_query_plans_with_SPPHints_total_actual_time = []
    # Construct the SQL command with the hints
    for query in queries:
        sql_command = SPPHints + '\n' + query
        executed_result = postgresql_workload_generator.execute_query(db_params, sql_command)
        executed_query_plans_with_SPPHints.append(executed_result)
        executed_query_plans_with_SPPHints_total_actual_time.append(executed_result['Actual Total Time'])
    # Execute the command using your database connection
    # connection.execute(sql_command)
    return executed_query_plans_with_SPPHints, executed_query_plans_with_SPPHints_total_actual_time

def Text_extraction(text, tables_index, tables_index_all):
    enc_table = [0] * len(tables_index)
    words = re.findall(r'\b\w+\b', text)
    for item in words:
        item = item.lower()
        if item in tables_index_all:
            enc_table[tables_index_all[item]] = 1
    return enc_table

def relation_extraction(text, tables_index, tables_index_all):
    relation_relation_enc = []
    for text_i in range(len(text)):
        relation_relation_enc.append(Text_extraction(text[text_i], tables_index, tables_index_all))
    return relation_relation_enc

def extract_hints_for_query(queries, SPPHints, tables_index, tables_index_all):
    new_queries = []

    hints_relation_enc = np.array(relation_extraction(SPPHints, tables_index, tables_index_all))
    queries_relation_enc = np.array(relation_extraction(queries, tables_index, tables_index_all))

    # Process each query.
    for query_i in range(len(queries)):
        # Get the query's relation encoding.
        query = queries[query_i]
        query_relation_enc_np = queries_relation_enc[query_i]
        # print(f"query_relation_enc_np: {query_relation_enc_np}")
        # If the query_relation_enc_np is 2D, combine along axis 0 (union of tokens)
        if query_relation_enc_np.ndim > 1:
            # For each column, if any value is 1, then mark that relation as present.
            query_relation_enc_np = (query_relation_enc_np.sum(axis=0) > 0).astype(int)

        # Now query_relation_enc_np should be 1D of shape (relation_num,)
        # For each hint (row), check: for every index i where hint_enc[i]==1,
        # query_relation_enc_np[i] should also be 1.
        matches = np.all(query_relation_enc_np >= hints_relation_enc, axis=1)
        matched_idxs = np.where(matches)[0]
        matched_hints = [SPPHints[i] for i in matched_idxs]
        if matched_hints:
            # Build the hint block in pg_hint_plan format (multiline).
            hint_block = "/*+\n" + "".join(matched_hints) + "*/"
        else:
            hint_block = ""
        # Append the hint block to the original query.
        new_query = hint_block + "\n" + query
        new_queries.append(new_query)
    return new_queries

def generate_subplan_pattern_hint(query_plans, explanation_results, dbname, f_min=0.5, f_max=0.5, cnt_min=10, ranking=4, max_hints_num=9999, max_relations=5, mode="all"):
    tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes = load_database_info(dbname)

    collect_operations = {}

    for i in tqdm(range(len(query_plans))):
        if 'Plans' not in query_plans[i][0]:
            continue
        for j in range(len(query_plans[i])):
            tree = replace_aliases_and_columns(query_plans[i][j], columns_list)
            subtrees = Data_augmentation([tree])
            L_pair_index, L_pair_label, _ = join_explanation_generate(tree)
            L_expl = explanation_results
            collect_operations = collect_operation_and_relations(subtrees, L_expl, L_pair_index, collect_operations, tables_index, tables_index_all, columns_index, attribute_range, nodes)

    new_collect_operations = update_collect_operations(collect_operations)
    print(f"Number of collect operations: {len(collect_operations)}")
    for key, value in new_collect_operations.items():
        if len(value) > 1:
            print(f"{key}: {value}")

    # Load the executed queries and their plans
    analyzed_collect_operations = analyze_collect_operations(new_collect_operations, f_min, f_max, cnt_min, ranking)
    print('\n----------------------------------------')
    print(f"Number of analyzed collect operations: {len(analyzed_collect_operations)}")

    nodes_inv = {v: k for k, v in nodes.items()}
    tables_index_inv = {v: k for k, v in tables_index.items()}

    SPPHints = generate_SPPHints(analyzed_collect_operations, nodes_inv, tables_index_inv)
    print('\n----------------------------------------')
    print(f"Number of all pg hints: {len(SPPHints)}")

    # Filter the pg hints based on the number of relations and mode
    filtered_SPPHints, filtered_hint_num = filter_SPPHints(SPPHints, max_hints_num, max_relations, mode)
    print('\n----------------------------------------')
    print(f"Number of filtered pg hints: {filtered_hint_num}")

    hints_save_path = f'../Results/{dbname}/hints/SPPHInts/f_min_{f_min}_f_max_{f_max}_cnt_min_{cnt_min}_ranking_{ranking}_max_hints_num_{max_hints_num}_max_relations_{max_relations}'
    os.makedirs(hints_save_path , exist_ok=True)
    with open(f'{hints_save_path}/SPPHints_mode_{mode}_{filtered_hint_num}_f_min_{f_min}_f_max_{f_max}_cnt_min_{cnt_min}_ranking_{ranking}_max_hints_num_{max_hints_num}_max_relations_{max_relations}.txt', 'w') as f:
        f.write(filtered_SPPHints)

def execute_query_with_SPPHints(queries, SPPHints, db_params):
    """
    Execute a query with the provided SPPHints.
    """
    tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes = load_database_info(db_params["dbname"])
    queries = extract_hints_for_query(queries, SPPHints, tables_index, tables_index_all)
    executed_query_plans_with_SPPHints = []
    executed_query_plans_with_SPPHints_total_actual_time = []
    # Construct the SQL command with the hints
    for query_i in range(len(queries)):
        query = queries[query_i]
        print(f'Executing query {query_i} with SPPHints')
        executed_result = postgresql_workload_generator.execute_query_with_hint(db_params, "LOAD 'pg_hint_plan';", query)
        executed_query_plans_with_SPPHints.append(executed_result)
        executed_query_plans_with_SPPHints_total_actual_time.append(executed_result['Actual Total Time'])
    return executed_query_plans_with_SPPHints, executed_query_plans_with_SPPHints_total_actual_time








