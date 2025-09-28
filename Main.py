# Import necessary libraries and modules
import pandas as pd
from SUBoost import AdaBoost, AdaC1, AdaCost, SUBoost, OUBoost, RUSBoost, SMOTEBoost, IMBoost
from process_folds import process_dataset
import warnings
warnings.filterwarnings("ignore")

def run_pipeline(base_names, path, model_name, model_class, num_iterations=50, **kwargs):
    """
    Run iterative evaluation for a given boosting model on multiple datasets.

    Parameters:
    - base_names: List of dataset base names (each represents a dataset to evaluate).
    - path: Directory path where datasets are located.
    - model_name: String name of the boosting model (for logging and file naming).
    - model_class: The class of the boosting model to instantiate and evaluate.
    - num_iterations: Number of repeated iterations for evaluation (default 50).
    - **kwargs: Additional keyword arguments to pass when initializing the model.

    This function runs multiple iterations of evaluation on each dataset, collects
    AUC (Area Under Curve) and G-Mean metrics, stores the results in a DataFrame,
    calculates averages, formats the results, and saves them to CSV files.
    """
    auc_results = {}      # Dictionary to store AUC values per dataset
    g_mean_results = {}   # Dictionary to store G-Mean values per dataset
    
    print("\n"+"# "+30*"-"+f" {model_name} "+"-"*30+" #")
    # Iterate over each dataset base name
    for base_name in base_names:
        auc_values = []      # List to accumulate AUC scores for multiple iterations
        g_mean_values = []   # List to accumulate G-Mean scores for multiple iterations

        # Perform evaluation for the defined number of iterations
        for _ in range(num_iterations):
            # process_folds is assumed to run cross-validation or similar evaluation
            # It returns mean AUC and G-Mean for the given model and dataset
            mean_auc, mean_g_mean = process_dataset(base_name, path, model_class, **kwargs)
            auc_values.append(mean_auc)
            g_mean_values.append(mean_g_mean)

        # Save iteration results per dataset
        auc_results[base_name] = auc_values
        g_mean_results[base_name] = g_mean_values

        # Log completion of processing for the current dataset and model
        print(f"{base_name} done")

    # Convert AUC results dictionary to a DataFrame with iterations as columns
    auc_df = pd.DataFrame.from_dict(
        auc_results, orient='index',
        columns=[f'Mean AUC_iter{i+1}' for i in range(num_iterations)]
    )
    # Add a column representing the average AUC across iterations for each dataset
    auc_df['Average'] = auc_df.mean(axis=1)
    # Add a row representing the average AUC across all datasets for each iteration
    auc_df.loc['Average'] = auc_df.mean(axis=0)
    # Format all AUC values as percentages with two decimal places
    auc_df = auc_df.applymap(lambda x: f"{x*100:.2f}")
    # Save AUC results to a CSV file named after the model
    auc_df.to_csv(f'{model_name.lower()}_auc_results.csv', index_label='dataset')

    # Repeat the above process for G-Mean results
    g_mean_df = pd.DataFrame.from_dict(
        g_mean_results, orient='index',
        columns=[f'Mean G-Mean_iter{i+1}' for i in range(num_iterations)]
    )
    g_mean_df['Average'] = g_mean_df.mean(axis=1)
    g_mean_df.loc['Average'] = g_mean_df.mean(axis=0)
    g_mean_df = g_mean_df.applymap(lambda x: f"{x*100:.2f}")
    g_mean_df.to_csv(f'{model_name.lower()}_g_mean_results.csv', index_label='dataset')
    
    print("\n"+30*"*")
    print("Average AUC :",auc_df.iloc[-1, -1])
    print("Average G-means :",g_mean_df.iloc[-1, -1])
    print(30*"*")

if __name__ == "__main__":
    # Dataset directory path
    path = ".\\Dataset\\"

    # List of dataset base filenames to evaluate
    base_names = [
        "glass1",
        "iris0",
        "haberman",
        "vehicle2",
        "vehicle1",
        "vehicle3",
        "glass-0-1-2-3_vs_4-5-6",
        "vehicle0",
        "segment0",
        "yeast-0-5-6-7-9_vs_4",
        "glass2",
        "glass4",
        "ecoli4",
        "page-blocks-1-3_vs_4",
        "abalone9-18",
        "dermatology-6",
        "zoo-3",
        "yeast-1-4-5-8_vs_7",
        "glass5",
        "yeast-2_vs_8",
        "lymphography-normal-fibrosis",
        "yeast4",
        "yeast-1-2-8-9_vs_7",
        "yeast5",
        "yeast6",
        "kddcup-land_vs_satan",
        "kddcup-rootkit-imap_vs_back"
    ]

    # Number of evaluation iterations per dataset
    num_iterations = 50

    # Dictionary defining boosting algorithms with their classes and default parameters
    boosting_algorithms = {
        "SUBoost": (SUBoost, {"n_estimators": 10, "depth": 5}),
        "RUSBoost": (RUSBoost, {"n_estimators": 10, "depth": 5}),
        "IMBoost": (IMBoost, {"n_estimators": 10, "depth": 5}),
        "OUBoost": (OUBoost, {"n_estimators": 10, "depth": 5}),
        "SMOTEBoost": (SMOTEBoost, {"n_estimators": 100, "depth": 10, "k_neighbors": 3}),
        "AdaBoost": (AdaBoost, {"n_estimators": 100, "depth": 10}),
        "AdaC1": (AdaC1, {"n_estimators": 100, "depth": 10, "cost_misclassifying_minority": 2.0}),
        "AdaCost": (AdaCost, {"n_estimators": 100, "depth": 10, "cost_misclassifying_minority": 2.0}),
    }

    # Run the evaluation pipeline for each boosting algorithm over all datasets
    for model_name, (model_class, params) in boosting_algorithms.items():
        run_pipeline(base_names, path, model_name, model_class, num_iterations=num_iterations, **params)
