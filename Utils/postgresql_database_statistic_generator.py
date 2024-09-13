import os
import string
from datetime import datetime
import numpy as np
import psycopg2
from gensim.models import Word2Vec

def is_float_num(v):
    try:
        float(v)
        return True
    except ValueError:
        return False

def postgresql_nodes_types():
    nodes = ['Seq Scan', 'Index Scan', 'Bitmap Index Scan', 'Bitmap Heap Scan', 'Index Only Scan', 'CTE Scan', 'Subquery Scan',
             'Hash', 'Hash Join', 'Merge Join', 'Nested Loop',
             'Sort', 'Incremental Sort', 'Aggregate', 'WindowAgg', 'Gather Merge', 'Gather', 'Group',
             'Unique', 'Memoize', 'Materialize', 'SetOp', 'Append', 'Merge Append', 'Result', 'Limit']
    return {node: index for index, node in enumerate(nodes)}

def postgresql_database_statistic_generator(db_params):
    # Get attributes of each table
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT pg_tables.tablename FROM pg_tables WHERE tablename NOT LIKE 'pg%' AND tablename NOT LIKE 'sql_%' ORDER BY tablename;")
    result = cursor.fetchall()
    tables = [table[0] for table in result]

    index = 0
    tables_index = {}
    tables_index_all = {}
    columns_list = []
    columns_index = {}
    table_columns_number = []
    attribute_range = {}

    for i, table in enumerate(tables):
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';")
        result = cursor.fetchall()
        tables_index[table] = i
        tables_index_all[table] = i
        table_columns_number.append(len(result))


        for (column_name, data_type) in result:
            full_column_name = f"{table}.{column_name}"
            tables_index_all[full_column_name] = i
            columns_list.append(column_name)
            columns_index[full_column_name] = index
            index += 1

            cursor.execute(f"SELECT {full_column_name} FROM {table};")
            result_c = cursor.fetchall()

            if data_type in ['integer', 'numeric', 'real', 'double precision', 'smallint', 'bigint']:
                # Extract and convert values to float where applicable
                filtered_values = [float(v[0].rstrip()) for v in result_c if v[0] is not None and is_float_num(v[0].rstrip())]
                if filtered_values:
                    value_set = set(filtered_values)
                    max_num = max(value_set)
                    min_num = min(value_set)
                    attribute_count = len(value_set)
                    try:
                        attri_num = 1 / attribute_count
                        # print(f"{column_name} {data_type}: {attri_num}")
                    except ZeroDivisionError:  # More specific error handling
                        print(f"Error in calculating attribute number for {column_name} of data_type {data_type}: Division by zero.")
                        attri_num = 0
                else:
                    min_num, max_num, attri_num = 0, 0, 0
                attribute_range[full_column_name] = [min_num, max_num, attri_num]

            elif data_type in ['date', 'time without time zone', 'time with time zone', 'timestamp without time zone', 'timestamp with time zone']:
                datetime_values = []
                for v in result_c:
                    if v[0]:
                        try:
                            datetime_values.append(datetime.strptime(v[0].rstrip(), '%Y-%m-%d'))
                        except:
                            datetime_values.append(datetime.strptime(v[0].rstrip(), '%Y-%m-%d %H:%M:%S'))
                if not datetime_values:
                    print(f"{data_type} type error: No valid date/time entries found.")
                    datetime_values = [datetime.min]
                datetime_set = set(datetime_values)
                max_datetime, min_datetime = max(datetime_set), min(datetime_set)
                try:
                    attri_num = 1 / len(datetime_set)
                    # print(f"{column_name} {data_type}: {attri_num}")
                except ZeroDivisionError:
                    print(f"Error in calculating attribute number for {column_name} of data_type {data_type}: Division by zero.")
                attribute_range[full_column_name] = [min_datetime, max_datetime, attri_num]

            elif data_type in ['character', 'character varying']:
                original_values = [v[0] for v in result_c if v[0] is not None]
                cleaned_values = []
                for value in original_values:
                    # Strip, lower, and remove punctuation in a single pass
                    cleaned = value.rstrip().lower().translate(str.maketrans('', '', string.punctuation))
                    cleaned_values.extend(cleaned.split())
                unique_values = set(original_values)
                attri_num = 1 / len(unique_values) if unique_values else 0  # Avoid division by zero
                # print(f"{column_name} {data_type}: {attri_num}")
                # Only proceed with Word2Vec if there are cleaned values
                if cleaned_values:
                    model = Word2Vec(sentences=[cleaned_values], vector_size=1, window=5, min_count=1, workers=8)
                    model_path = f'Data/Word2vec/{table}_{column_name}.model'
                    model.save(model_path)
                else:
                    model_path = 'None'
                attribute_range[full_column_name] = [model_path, attri_num]

            else:
                print(f"Data type {data_type} not supported for column {full_column_name}.")
                pass

    save_path = 'Data/' + db_params['dbname'] + '/database_statistic'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "/tables_index", tables_index)
    np.save(save_path + "/tables_index_all", tables_index_all)
    np.save(save_path + "/columns_index", columns_index)
    np.save(save_path + "/attribute_range", attribute_range)
    np.save(save_path + "/table_columns_number", table_columns_number)
    np.save(save_path + "/columns_list", list(set(columns_list)))
    np.save(save_path + "/postgresql_nodestypes_all", postgresql_nodes_types())

def generate_postgresql_database_statistic(db_params):
    postgresql_database_statistic_generator(db_params)
