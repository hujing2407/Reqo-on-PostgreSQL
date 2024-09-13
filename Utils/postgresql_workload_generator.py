import os
import sys
import numpy as np
import psycopg2

hints_off = ['set enable_nestloop = off;',
             'set enable_nestloop = off; set enable_indexscan = off;',
             'set enable_hashjoin = off;',
             'set enable_hashjoin = off; set enable_indexscan = off;',
             'set enable_mergejoin = off;',
             'set enable_mergejoin = off; set enable_indexscan = off;',
             'set enable_nestloop = off; set enable_mergejoin = off;',
             'set enable_nestloop = off; set enable_mergejoin = off; set enable_indexscan = off;',
             'set enable_nestloop = off; set enable_hashjoin = off;',
             'set enable_nestloop = off; set enable_hashjoin = off; set enable_indexscan = off;',
             'set enable_mergejoin = off; set enable_hashjoin = off;',
             'set enable_mergejoin = off; set enable_hashjoin = off; set enable_indexscan = off;']

def load_query_for_analyze(path_name):
    with open(path_name, "r") as f:
        # Strip newline characters and skip empty lines directly in the comprehension
        queries = [f"explain (ANALYZE true, FORMAT JSON) {line.strip()}" for line in f if line.strip()]
    return queries

def load_query_for_explain(path_name):
    with open(path_name, "r") as f:
        # Strip newline characters and skip empty lines directly in the comprehension
        queries = [f"explain (ANALYZE false, FORMAT JSON) {line.strip()}" for line in f if line.strip()]
    return queries

def execute_query(db_params, q):
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(q)  # Execute query
                result = cursor.fetchall()[0][0][0]['Plan']
    except Exception as e:
        print(f"An error occurred while executing the query: {e}")
        sys.exit()
    return result

def execute_query_with_hint(db_params, h, q):
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(h)  # Execute hint
                cursor.execute(q)  # Execute query
                result = cursor.fetchall()[0][0][0]['Plan']
    except Exception as e:
        print(f"An error occurred while executing the query: {e}")
        sys.exit()
    return result

def generate_postgresql_workload_with_hints(db_params, query_file_path):
    queries = load_query_for_analyze(query_file_path)
    total_queries_executed = 0
    total_query_plans_executed = 0
    total_timeouts = 0
    executed_query_plans = []
    executed_query_indices = []
    executed_query_plan_indices = []

    for i, q in enumerate(queries):
        query_plans = []
        plan_indices = []
        timeouts = 0
        for h_index in range(len(hints_off) + 1):
            try:
                print(f'Running query {i}_{h_index}')
                if h_index == 0:
                    plan = execute_query(db_params, q)
                else:
                    plan = execute_query_with_hint(db_params, hints_off[h_index - 1], q)
                query_plans.append(plan)
                plan_indices.append(h_index)
            except Exception as e:
                print(f'Timeout or error on query {i}_{h_index}: {e}')
                timeouts += 1
                continue

        if len(query_plans) >= 2:
            executed_query_plans.append(query_plans)
            executed_query_indices.append(i)
            executed_query_plan_indices.append(plan_indices)
            total_queries_executed += 1
            total_query_plans_executed += len(query_plans)
            total_timeouts += timeouts

    # Save results
    assert len(executed_query_plans) == len(executed_query_plan_indices)
    dbname = db_params["dbname"]
    save_path = f'Data/{dbname}/workloads/'
    os.makedirs(save_path, exist_ok=True)

    file_base = f'{save_path}postgresql_{dbname}_executed_query'

    np.save(f'{file_base}_plans.npy', np.array(executed_query_plans, dtype=object))
    np.save(f'{file_base}_index.npy', np.array(executed_query_indices, dtype=object))
    np.save(f'{file_base}_plans_index.npy', np.array(executed_query_plan_indices, dtype=object))

    print(f'A total of {total_queries_executed} query plans with hints were executed')
    print(f'A total of {total_query_plans_executed} query plans were executed')
    print(f'A total of {total_timeouts} queries timed out')

def generate_postgresql_workload(db_params, query_file_path):
    queries = load_query_for_analyze(query_file_path)
    total_queries_executed = 0
    total_timeouts = 0
    executed_query_plans = []
    executed_query_indices = []

    for i, q in enumerate(queries):
        try:
            executed_query_plans.append(execute_query(db_params, q))
            executed_query_indices.append(i)
            total_queries_executed += 1
        except Exception as e:
            print(f'Timeout or error on query {i}: {e}')
            total_timeouts += 1
            continue

    # Save results
    dbname = db_params["dbname"]
    save_path = f'Data/{dbname}/workloads/'
    os.makedirs(save_path, exist_ok=True)

    file_base = f'{save_path}postgresql_{dbname}_executed_query'

    np.save(f'{file_base}_plans.npy', np.array(executed_query_plans, dtype=object))
    np.save(f'{file_base}_index.npy', np.array(executed_query_indices, dtype=object))

    print(f'A total of {total_queries_executed} query plans were executed')
    print(f'A total of {total_timeouts} queries timed out')
