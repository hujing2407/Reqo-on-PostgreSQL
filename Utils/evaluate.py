import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean, median
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.stats import spearmanr

def get_qerror_and_spearman(outputs, targets, max_label, min_label):
    outputs = np.exp(outputs * (max_label - min_label) + min_label) - 1
    qerror = np.maximum(outputs / targets, targets / outputs)
    spearman_corr = spearmanr(outputs, targets, nan_policy='omit')[0]
    qerror.sort()

    num_plans = len(qerror)

    percentiles = [50, 90, 99]
    percentile_values = np.percentile(qerror, percentiles)

    cost_estimation_results = [
        np.round(np.mean(qerror[:int(0.5 * num_plans)]), 3), # Top 50% mean q-error
        np.round(np.mean(qerror[:int(0.9 * num_plans)]), 3), # Top 90% mean q-error
        np.round(np.mean(qerror[:int(0.99 * num_plans)]), 3), # Top 99% mean q-error
        np.round(np.mean(qerror), 3), # Mean q-error
        np.round(percentile_values[0], 3),  # 50th percentile q-error
        np.round(percentile_values[1], 3),  # 90th percentile q-error
        np.round(percentile_values[2], 3),  # 99th percentile q-error
        np.round(qerror[-1], 3),  # Max q-error
        np.round(spearman_corr, 3) # Spearman correlation
    ]

    return cost_estimation_results

def get_plansubop_and_runtime(outputs, targets, query_postgres_cost, query_plans_index_num):
    query_num = len(query_plans_index_num)
    postgres_select_runtime = []
    model_select_runtime = []
    optimal_runtime = []
    plan_subop = []
    p_n = 0
    total_postgres_select_runtime = 0
    total_model_select_runtime = 0
    total_optimal_runtime = 0
    imp_runtime_num = 0
    reg_runtime_num = 0
    for i in range(query_num):
        query_actual_set = targets[p_n:p_n + query_plans_index_num[i]]
        query_pred_set = outputs[p_n:p_n + query_plans_index_num[i]]

        postgres_select_index = np.argmin(query_postgres_cost[i])
        postgres_select_runtime_i = query_actual_set[postgres_select_index]

        model_select_index = np.argmin(query_pred_set)
        model_select_runtime_i = query_actual_set[model_select_index]

        fastest_actual_runtime_i = np.min(query_actual_set)

        plan_subop.append(model_select_runtime_i / fastest_actual_runtime_i)

        postgres_select_runtime.append(postgres_select_runtime_i)
        model_select_runtime.append(model_select_runtime_i)
        optimal_runtime.append(fastest_actual_runtime_i)

        total_model_select_runtime += model_select_runtime_i
        total_postgres_select_runtime += postgres_select_runtime_i
        total_optimal_runtime += fastest_actual_runtime_i

        if model_select_runtime_i < postgres_select_runtime_i:
            imp_runtime_num += 1
        elif model_select_runtime_i > postgres_select_runtime_i:
            reg_runtime_num += 1

        p_n += query_plans_index_num[i]

    plan_subop.sort()
    robustness_results = [
        round(np.mean(plan_subop[:int(0.5 * len(plan_subop))]), 3), # Top 50% mean plan suboptimality
        round(np.mean(plan_subop[:int(0.90 * len(plan_subop))]), 3), # Top 90% mean plan suboptimality
        round(np.mean(plan_subop[:int(0.99 * len(plan_subop))]), 3), # Top 99% mean plan suboptimality
        round(np.mean(plan_subop), 3), # Mean plan suboptimality
        round(np.median(plan_subop), 3), # 50th percentile plan suboptimality
        round(plan_subop[int(0.90 * len(plan_subop))], 3), # 90th percentile plan suboptimality
        round(plan_subop[int(0.99 * len(plan_subop))], 3), # 99th percentile plan suboptimality
        round(np.max(plan_subop), 3), # Max plan suboptimality
        round(imp_runtime_num / len(query_plans_index_num), 3), # Improved runtime percentage compared to Postgres
        round(reg_runtime_num / len(query_plans_index_num), 3), # Regressed runtime percentage compared to Postgres
        round(total_model_select_runtime / total_postgres_select_runtime, 3), # Total model select runtime / total postgres select runtime
        round(total_model_select_runtime / total_optimal_runtime, 3), # Total model select runtime / total optimal runtime
        round(total_postgres_select_runtime / total_optimal_runtime, 3) # Total postgres select runtime / total optimal runtime
    ]

    return robustness_results, [postgres_select_runtime, model_select_runtime, optimal_runtime]


def write_results_to_file(results, expl_or_not=False, filename='results.txt'):
    # Unpack results (assuming each value is in the correct position based on the provided structure)
    qerror_metrics = results[0:9]  # First nine are q-error and Spearman correlation
    robustness_metrics = results[9:22]  # Remaining are robustness and runtime metrics
    if expl_or_not:
        expl_metrics = results[22:]

    with open(filename, 'w') as file:
        file.write("Cost Estimation Results:\n")
        file.write("-" * 30 + "\n")
        file.write(f"Mean Q-Error (Top 50%): {qerror_metrics[0]}\n")
        file.write(f"Mean Q-Error (Top 90%): {qerror_metrics[1]}\n")
        file.write(f"Mean Q-Error (Top 99%): {qerror_metrics[2]}\n")
        file.write(f"Overall Mean Q-Error: {qerror_metrics[3]}\n")
        file.write(f"50th Percentile Q-Error: {qerror_metrics[4]}\n")
        file.write(f"90th Percentile Q-Error: {qerror_metrics[5]}\n")
        file.write(f"99th Percentile Q-Error: {qerror_metrics[6]}\n")
        file.write(f"Maximum Q-Error: {qerror_metrics[7]}\n")
        file.write(f"Spearman Correlation: {qerror_metrics[8]}\n")

        file.write("\nRobustness Results:\n")
        file.write("-" * 30 + "\n")
        file.write(f"Mean Plan Suboptimality (Top 50%): {robustness_metrics[0]}\n")
        file.write(f"Mean Plan Suboptimality (Top 90%): {robustness_metrics[1]}\n")
        file.write(f"Mean Plan Suboptimality (Top 99%): {robustness_metrics[2]}\n")
        file.write(f"Overall Mean Plan Suboptimality: {robustness_metrics[3]}\n")
        file.write(f"50th Percentile Plan Suboptimality: {robustness_metrics[4]}\n")
        file.write(f"90th Percentile Plan Suboptimality: {robustness_metrics[5]}\n")
        file.write(f"99th Percentile Plan Suboptimality: {robustness_metrics[6]}\n")
        file.write(f"Maximum Plan Suboptimality: {robustness_metrics[7]}\n")
        file.write(f"Improved Runtime Percentage: {robustness_metrics[8]}\n")
        file.write(f"Regressed Runtime Percentage: {robustness_metrics[9]}\n")
        file.write(f"Model vs. PostgreSQL Runtime Ratio: {robustness_metrics[10]}\n")
        file.write(f"Model vs. Optimal Runtime Ratio: {robustness_metrics[11]}\n")
        file.write(f"PostgreSQL vs. Optimal Runtime Ratio: {robustness_metrics[12]}\n")

        if expl_or_not:
            file.write("\nExplanation Results:\n")
            file.write("-" * 30 + "\n")
            file.write(f"Top1 Accuracy: {expl_metrics[0]}\n")
            file.write(f"Top1and2 Accuracy: {expl_metrics[1]}\n")
            file.write(f"Top1or2 Accuracy: {expl_metrics[2]}\n")
            file.write(f"Top1 Subinfluence Ratio: {expl_metrics[3]}\n")
            file.write(f"Top1and2 Subinfluence Ratio: {expl_metrics[4]}\n")

def plot_runtimes(postgres_runtimes, reqo_runtimes, optimal_runtimes, save_path='runtime_performance.png'):
    # Prepare the cumulative sums for each runtime list
    postgres_cumulative = np.cumsum(postgres_runtimes) / 1000
    reqo_cumulative = np.cumsum(reqo_runtimes) / 1000
    optimal_cumulative = np.cumsum(optimal_runtimes) / 1000

    # Number of queries
    queries = list(range(1, len(postgres_runtimes) + 1))
    plt.rcParams.update({'font.size': 24})
    # Creating the plot
    plt.figure(figsize=(16, 12))
    plt.plot(queries, postgres_cumulative, label='PostgresSQL', linestyle='-', color='blue')
    plt.plot(queries, reqo_cumulative, label='Reqo', linestyle='-', color='green')
    plt.plot(queries, optimal_cumulative, label='Optimal', linestyle='-', color='red')

    # Adding titles and labels
    plt.title(f'Runtime Performace Comparison')
    plt.xlabel('Query Number')
    plt.ylabel('Total Runtime (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def get_explanation_results(pred_expl, expl_labels, testset_index, subtree_labels_load, subtree_join_pair_index):
    join_num = 0
    top1_acc, top1and2_acc, top1or2_acc = 0, 0, 0
    top1_subinfl_ratio, top1and2_subinfl_ratio = 0, 0

    for s in testset_index:
        subtree_join_num = len(subtree_labels_load[s])
        pred_expl_s = pred_expl[join_num:join_num + subtree_join_num]
        label_expl_s = expl_labels[join_num:join_num + subtree_join_num]
        join_pairs = subtree_join_pair_index[s]

        pred_scores, label_scores = [], []

        for left, right in join_pairs:
            if isinstance(right, list): # nodes with more than one child
                right_sum_pred = sum(pred_expl_s[j] for j in right)
                right_sum_label = sum(label_expl_s[j] for j in right)
            else:
                if left == right: # leaf nodes
                    right_sum_pred = 0
                    right_sum_label = 0
                else: # nodes with only one child
                    right_sum_pred = pred_expl_s[right]
                    right_sum_label = label_expl_s[right]

            pred_scores.append(pred_expl_s[left] - right_sum_pred)
            label_scores.append(label_expl_s[left] - right_sum_label)

        pred_sort_idx = np.argsort(pred_scores)
        label_sort_idx = np.argsort(label_scores)

        if len(pred_sort_idx) != len(label_sort_idx):
            print("Error: Inconsistent length between prediction and label indices.")
            continue

        # Check top1 and top1&2 accuracy
        top1_match = pred_sort_idx[-1] == label_sort_idx[-1]
        top1and2_match = (len(pred_sort_idx) > 1 and top1_match and pred_sort_idx[-2] == label_sort_idx[-2]) or (len(pred_sort_idx) == 1 and top1_match)

        if top1_match:
            top1_acc += 1
        if top1and2_match:
            top1and2_acc += 1
        if pred_sort_idx[-1] in label_sort_idx[-2:] or pred_sort_idx[-2] in label_sort_idx[-2:]:
            top1or2_acc += 1

        # Calculate subgraph subinfluence ratios
        top1_subinfl_ratio += label_scores[pred_sort_idx[-1]] / label_scores[label_sort_idx[-1]]

        if len(pred_sort_idx) > 1:
            top1and2_subinfl_ratio += (label_scores[pred_sort_idx[-1]] + label_scores[pred_sort_idx[-2]]) / \
                                   (label_scores[label_sort_idx[-1]] + label_scores[label_sort_idx[-2]])
        else:
            top1and2_subinfl_ratio += label_scores[pred_sort_idx[-1]] / label_scores[label_sort_idx[-1]]

        join_num += subtree_join_num

    num_samples = len(testset_index)
    return [
        round(top1_acc / num_samples, 3), # Model accuracy in identifying the Top1 most influential subgraph.
        round(top1and2_acc / num_samples, 3), # Model accuracy in identifying the Top1 and 2 most influential subgraphs.
        round(top1or2_acc / num_samples, 3), # Model accuracy in identifying the Top1 or 2 most influential subgraphs.
        round(top1_subinfl_ratio / num_samples, 3), # The ratio of model selected Top 1 influenced subgrah's latency to the actual Top 1 influenced subgraph's latency.
        round(top1and2_subinfl_ratio / num_samples, 3) # The ratio of model selected Top 1 and 2 influenced subgraphs' latency to the actual Top 1 and 2 influenced subgraphs' latency.
    ]

def plot_explanation(model_results, postgres_cost, total_plan_num,  subtree_labels_load, subtree_join_pair_index, save_path):
    dataset_index = list(range(total_plan_num))
    expl_labels = np.concatenate(subtree_labels_load)
    postgres_explanation_results = get_explanation_results(postgres_cost, expl_labels, dataset_index, subtree_labels_load, subtree_join_pair_index)
    model_explanation_results = model_results[22:]

    metrics = ['top1_acc', 'top1and2_acc', 'top1or2_acc', 'top1_subinfl_ratio', 'top1and2_subinfl_ratio']

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, postgres_explanation_results, width, label='PostgreSQL')
    rects2 = ax.bar(x + width / 2, model_explanation_results, width, label='Reqo')

    ax.set_ylabel('Values')
    ax.set_title('PostgreSQL vs Reqo Explanation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()

    def add_values(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_values(rects1)
    add_values(rects2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

