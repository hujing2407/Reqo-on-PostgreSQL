# Reqo: A Comprehensive Learning-Based Cost Model for Robust and Explainable Query Optimization

## Introduction
Query optimizers are critical for relational database performance. To this end, we introduce Reqo, a novel cost model for query optimization that employs Bidirectional Graph Neural Networks (Bi-GNN) combined with Gated Recurrent Units (GRU) as tree models, integrates a learning-to-rank uncertainty-aware cost estimator, and, for the first time, incorporates explainability techniques into learning-based cost models and generates hints for optimizing future plan generation. Reqo improves the performance of cost models in three dimensions: cost estimation accuracy, plan selection robustness and cost model explainability. The repository contains the code for the naive prototype of Reqo based on PostgreSQL.

## PostgreSQL Setup
This prototype is based on PostgreSQL 15.1. Reqo requires configuring a PostgreSQL database to generate the necessary workloads.

### Step 1: Install PostgreSQL
First, install PostgreSQL on your system. The installation process varies depending on your operating system. Detailed installation instructions of PostgreSQL can be found in the [PostgreSQL official documentation](https://www.postgresql.org/download/).
```
sudo apt update
sudo apt install postgresql postgresql-contrib
```

### Step 2: Create a Database
Once PostgreSQL is installed, ensure the server is running and create a new database to generate the required workloads. For instance, using the STATS-CEB database setup, as detailed in the [End-to-End CardEst Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark).
1. **Download STATS database:**
```
git clone https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark.git
cd End-to-End-CardEst-Benchmark
```

2. **Load STATS database to PostgreSQL:**
```
sudo -u postgres psql
CREATE DATABASE stats;
\c stats
\i datasets/stats_simplified/stats.sql
\i scripts/sql/stats_load.sql
\i scripts/sql/stats_index.sql
```

## Model Setup
### Step 1: Install Reqo
1. **Clone the repository:**
```
git clone https://github.com/BaomingChang/Reqo-on-PostgreSQL.git
```

2. **Install required Python packages:**
```
cd Reqo-on-PostgreSQL
pip install -r requirements.txt
```

### Step 2: Run Reqo
The prototype can be run by executing the command below. This will generate the necessary workloads and datasets, and then train and evaluate the model once the requisite PostgreSQL server configurations, the location of the query file, and specific model preferences are provided.
```bash
python main.py --dbname your_database_name --user your_username --password your_password --host your_host --port your_port --query_file_path path_to_your_query_file --explain_or_not True_or_False --save_model True_or_False
```
Options
* `--dbname`: Specify the PostgreSQL database name.
* `--user`: PostgreSQL username (default: postgres).
* `--password`: Password for the PostgreSQL user.
* `--host`: Host address of the PostgreSQL server (default: localhost).
* `--port: Port number where PostgreSQL is running (default: 5432).
* `--query_file_path`: Path to the SQL query file.
* `--explain_or_not`: Choose the two modes of the model:
	* `False`: Employ model without explainability technique for faster training.
	* `True`: Employ the explainability technique to enhance the model's ability to explain how different subgraphs of the query plan contribute to the final predicted cost, enhancing overall explainability but reducing training speed.
* `--save_model`: Whether to save the trained model (True or False).

More parameters for model training can be modified directly in the main.py file. During this process, each query is executed with 12 hints, and the generated database statistics, workloads, and datasets will be stored in a folder named after the using database within the `Data` directory. A CUDA-compatible GPU is recommended to leverage accelerated computing capabilities.

### Step 3: Evaluate Reqo
After training, the application employs k-fold cross-validation by default. For each fold, the trained model and its evaluation results on the test set are saved under the `Results` directory, organized into subfolders named after the database. The average evaluation results across all folds are also calculated and stored.

Additionally, comparative visualizations of runtime and explainability metrics against PostgreSQL are generated, as demonstrated in the STATS example below:

<p align="center">
  <img src="/Results/stats/reqo_with_explanation_runtime_performance.png" alt="Runtime performance (PostgreSQL vs Reqo vs Optimal)" width="49%"/>
  <img src="/Results/stats/reqo_with_explanation_explanation_performance.png" alt="Explanation performance (PostgreSQL vs Reqo)" width="49%"/>
</p>

