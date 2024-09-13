import os
import numpy as np
from numpy import nanmean
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Models.reqo_model_with_explanation import Reqo
from Utils.loss import *
from Utils.evaluate import get_qerror_and_spearman, get_plansubop_and_runtime, write_results_to_file, plot_runtimes, get_explanation_results, plot_explanation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(dataset, batch_size, shuffle_or_not):
    data = [Data(x=torch.FloatTensor(row[0]),
                 edge_index=sort_edge_index(torch.LongTensor(row[1])),
                 y=row[2]) for row in dataset]
    dataset_loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle_or_not)
    return dataset_loader


def train_with_explanation(dbname, reqo_config, k_i, trainset, testset, save_path, query_plans_index_num_i, query_postgres_cost_i,
                           subtree_num_load_i, testset_index, subtree_index_load, subtree_labels_load, subtree_join_pair_index, save_model):
    batch_size = reqo_config["batch_size"]
    table_columns_number = torch.tensor(np.load(f'Data/{dbname}/database_statistics/table_columns_number.npy')).to(device)

    testset_size = len(testset)
    testset_subtree_size = sum(subtree_num_load_i)
    train_loader = load_dataset(trainset, batch_size, True)
    test_loader = load_dataset(testset, batch_size, False)

    subtree_labels_load_flatten = [item for sublist in subtree_labels_load for item in sublist]
    max_label_log = math.log(max(subtree_labels_load_flatten) + 1)
    min_label_log = math.log(min(subtree_labels_load_flatten) + 1)

    encoder_params = {k: v for k, v in reqo_config.items() if k.startswith("encoder_")}
    estimator_params = {k: v for k, v in reqo_config.items() if k.startswith("estimator_")}
    explainer_params = {k: v for k, v in reqo_config.items() if k.startswith("explainer_")}
    encoder_params["encoder_table_num"] = len(table_columns_number)
    encoder_params["encoder_column_num"] = sum(table_columns_number)

    model = Reqo(encoder_params=encoder_params, estimator_params=estimator_params, explainer_params=explainer_params)
    model = model.to(device)

    criteon_data_uncertainty = DataUncertaintyLoss()
    criteon_ranking = PairRankingLoss()
    criteon_explanation = ExplanationLoss()
    optimizer = optim.Adam(model.parameters(), lr=reqo_config["learning_rate"])

    subtree_index = [torch.LongTensor(item) for item in subtree_index_load]
    subtree_labels = [torch.FloatTensor(item) for item in subtree_labels_load]

    epochs = 100
    early_stop = 0
    best_test_perf =float('inf')
    for epoch in range(epochs):
        if early_stop >= 20:
            break
        # Train
        model.train()
        step = 0
        train_loss_all = 0
        with tqdm(train_loader, desc=f'Fold {k_i} Epoch: {epoch + 1} Training ', unit='batch') as train_loader:
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                batch_train_pred, batch_train_va, batch_train_iv, batch_train_expl, batch_train_global_labels, batch_train_local_labels = model(batch, table_columns_number, subtree_index, subtree_labels)
                train_data_uncertainty_loss = criteon_data_uncertainty(batch_train_pred, batch_train_va, batch_train_global_labels, max_label_log, min_label_log)
                train_ranking_loss = criteon_ranking(batch_train_iv, batch_train_global_labels, max_label_log, min_label_log)
                train_explanation_loss = criteon_explanation(batch_train_expl, batch_train_local_labels)
                train_loss = train_data_uncertainty_loss + train_ranking_loss + train_explanation_loss
                train_loss.backward()
                optimizer.step()

                batch_graph_num = batch.num_graphs
                train_loss_all += train_loss.item()*batch_graph_num
                step += batch_graph_num
            avg_train_loss = float(train_loss_all/step)

        # Test
        model.eval()
        step = 0
        test_loss_all = 0
        actual_latency = torch.zeros(testset_size, device=device)
        pred_ev = torch.zeros(testset_size, device=device)
        pred_va = torch.zeros(testset_size, device=device)
        pred_iv = torch.zeros(testset_size, device=device)
        pred_expl = torch.zeros(testset_subtree_size, device=device)
        expl_labels = torch.zeros(testset_subtree_size, device=device)
        subtree_n = 0
        model.eval()
        torch.cuda.empty_cache()
        with tqdm(test_loader, desc=f'Fold {k_i} Epoch: {epoch + 1} Testing ', unit='batch') as test_loader:
            for batch in test_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    batch_test_pred, batch_test_va, batch_test_iv, batch_test_expl, batch_test_global_labels, batch_test_local_labels = model(batch, table_columns_number, subtree_index, subtree_labels)
                test_data_uncertainty = criteon_data_uncertainty(batch_test_pred, batch_test_va, batch_test_global_labels, max_label_log, min_label_log)
                test_ranking_loss = criteon_ranking(batch_test_iv, batch_test_global_labels, max_label_log, min_label_log)
                test_explanation_loss = criteon_explanation(batch_test_expl, batch_test_local_labels)
                test_loss = test_data_uncertainty + test_ranking_loss + test_explanation_loss
                batch_graph_num = batch.num_graphs
                test_loss_all += test_loss.item()*batch_graph_num
                end = step + batch_graph_num
                pred_ev[step:end] = batch_test_pred.view(-1)
                pred_va[step:end] = batch_test_va.view(-1)
                pred_iv[step:end] = batch_test_iv.view(-1)
                pred_expl[subtree_n:subtree_n + batch_test_expl.shape[0]] = batch_test_expl.view(-1)
                expl_labels[subtree_n:subtree_n + batch_test_local_labels.shape[0]] = batch_test_local_labels.view(-1)
                actual_latency[step:end] = batch_test_global_labels.view(-1)
                step = end
                subtree_n += batch_test_expl.shape[0]
            avg_test_loss = float(test_loss_all/step)

        pred_ev = pred_ev.cpu().numpy()
        pred_iv = pred_iv.cpu().numpy()
        actual_latency = actual_latency.cpu().numpy()
        pred_expl = pred_expl.cpu().numpy()
        expl_labels = expl_labels.cpu().numpy()

        cost_estimation_results = get_qerror_and_spearman(pred_ev, actual_latency, max_label_log, min_label_log)
        robustness_results, runtime_per_query = get_plansubop_and_runtime(pred_iv, actual_latency, query_postgres_cost_i, query_plans_index_num_i)
        explanation_results = get_explanation_results(pred_expl, expl_labels, testset_index, subtree_labels_load, subtree_join_pair_index)
        print(f'Fold {k_i} Epoch {epoch + 1}: train_loss: {avg_train_loss}, test_loss: {avg_test_loss}, spearmancorrelation: {cost_estimation_results[-1]}, optimal_runtime_ratio: {robustness_results[11]}, Top1and2_explanation_accuracy: {explanation_results[1]}')

        # Early stop based on test loss
        if avg_test_loss < best_test_perf:
            best_test_perf = avg_test_loss
            best_model = model.state_dict()
            best_cost_estimation_results = cost_estimation_results
            best_robustness_results = robustness_results
            best_runtime_per_query = runtime_per_query
            best_explanation_results = explanation_results
            early_stop = 0
        else:
            early_stop += 1

        # # Early stop based on optimal runtime ratio
        # if robustness_results[11] < best_test_perf:
        #     best_test_perf = robustness_results[11]
        #     best_model = model.state_dict()
        #     best_cost_estimation_results = cost_estimation_results
        #     best_robustness_results = robustness_results
        #     best_runtime_per_query = runtime_per_query
        #     best_explanation_results = explanation_results
        #     early_stop = 0
        # else:
        #     early_stop += 1

    cost_estimation_results = best_cost_estimation_results
    robustness_results = best_robustness_results
    runtime_per_query = best_runtime_per_query
    explanation_results = best_explanation_results

    print(f'Fold {k_i}: test results: qerror_median: {cost_estimation_results[4]} qerror_top99mean: {cost_estimation_results[2]}, spearman_correlation: {cost_estimation_results[-1]}, subop_median: {robustness_results[4]}, subop_top99mean: {robustness_results[2]}, model_to_postgresql_runtime_ratio: {robustness_results[10]}, model_to_optimal_runtime_ratio: {robustness_results[11]}, Top1and2_explanation_accuracy: {explanation_results[1]}, Top1and2_explanation_subinfl: {explanation_results[4]}')
    os.makedirs(save_path, exist_ok=True)
    write_results_to_file(cost_estimation_results + robustness_results + explanation_results, expl_or_not=True, filename=save_path + 'reqo_with_explanation_fold_' + str(k_i) + '_results.txt')
    if save_model:
        torch.save(best_model, save_path + 'reqo_with_explanation_fold_' + str(k_i) + '_model.pth')
    return cost_estimation_results + robustness_results + explanation_results, runtime_per_query


def k_fold_train_with_explanation(dbname, reqo_config, k=10, save_model=False):
    save_path = f'Results/{dbname}/'
    os.makedirs(save_path, exist_ok=True)
    # Load data of executed query plans
    dataset = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_dataset_with_explanation.npy', allow_pickle=True)
    query_plans_index_num = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_index_num_with_explanation.npy', allow_pickle=True)
    query_postgres_cost = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_postgres_cost_with_explanation.npy', allow_pickle=True)
    # Load data for explanation
    subtree_num_load = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_subtrees_num_with_explanation.npy',allow_pickle=True)
    subtree_index_load = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_sample_index_with_explanation.npy', allow_pickle=True)
    subtree_labels_load = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_subtree_labels_with_explanation.npy', allow_pickle=True)
    subtree_join_pair_index = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_join_pair_index_for_explain_with_explanation.npy', allow_pickle=True)
    subtree_postgres_cost = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_subtree_postgres_cost_with_explanation.npy', allow_pickle=True)
    k_sample_num = round(len(query_plans_index_num) / k)

    all_results = []
    all_postgres_runtimes, all_reqo_runtimes, all_optimal_runtimes = [], [], []
    for k_i in range(0, k):
        sample_q_num1 = k_sample_num*k_i
        sample_q_num2 = k_sample_num*(k_i+1)
        sample_p_num1 = sum(query_plans_index_num[:sample_q_num1])
        sample_p_num2 = sample_p_num1 + sum(query_plans_index_num[sample_q_num1:sample_q_num2])

        trainset = np.concatenate((dataset[:sample_p_num1], dataset[sample_p_num2:]), axis=0)
        testset = dataset[sample_p_num1:sample_p_num2]

        query_plans_index_num_i = query_plans_index_num[sample_q_num1:sample_q_num2]
        query_postgres_cost_i = query_postgres_cost[sample_q_num1:sample_q_num2]
        subtree_num_load_i = subtree_num_load[sample_p_num1:sample_p_num2]
        testset_index = [i for i in range(sample_p_num1, sample_p_num2)]

        results, runtime_per_query = train_with_explanation(dbname, reqo_config, k_i+1, trainset, testset, save_path+'fold_'+str(k_i+1)+'/', query_plans_index_num_i, query_postgres_cost_i,
                                                            subtree_num_load_i, testset_index, subtree_index_load, subtree_labels_load, subtree_join_pair_index, save_model)
        all_results.append(results)
        all_postgres_runtimes.extend(runtime_per_query[0])
        all_reqo_runtimes.extend(runtime_per_query[1])
        all_optimal_runtimes.extend(runtime_per_query[2])

    avg_results = nanmean(np.array(all_results), axis=0)
    write_results_to_file(avg_results, expl_or_not=True, filename=save_path + 'reqo_with_explanation_avg_results.txt')
    plot_runtimes(all_postgres_runtimes, all_reqo_runtimes, all_optimal_runtimes, save_path + 'reqo_with_explanation_runtime_performance.png')
    plot_explanation(avg_results, subtree_postgres_cost, sum(query_plans_index_num), subtree_labels_load, subtree_join_pair_index, save_path + 'reqo_with_explanation_explanation_performance.png')


if __name__ == "__main__":
    dbname = 'stats'
    reqo_config = {'batch_size': 256, 'learning_rate': 0.001,
              'encoder_attention_heads': 8, 'encoder_conv_layers': 4, 'encoder_gnn_embedding_dim': 256, 'encoder_gnn_dropout_rate': 0.1, 'encoder_dirgnn_alpha': 0.3, 'encoder_node_type_embedding_dim': 16, 'encoder_column_embedding_dim': 8,
              'explainer_fcn_layers': 4, 'explainer_explanation_embedding_dim': 512, 'explainer_fcn_dropout_rate': 0.1,
              'estimator_fcn_layers': 4, 'estimator_estimation_embedding_dim': 512, 'estimator_fcn_dropout_rate': 0.1}
    k_fold_train_with_explanation(dbname, reqo_config, k=10)
