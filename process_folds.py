# Process folds
import glob
import numpy as np
from load_dataset import load_data
from calculate_metrics import calculate_metrics


def process_dataset(base_name, path, model_class, **kwargs):
    """Process one dataset (all folds) and return mean AUC and G-Mean."""
    auc_scores = []  # List to store AUC scores for each fold
    g_mean_scores = []  # List to store G-Mean scores for each fold

    # Get sorted lists of all training and testing files matching the dataset base name
    train_files = sorted(glob.glob(f"{path}{base_name}*tra.dat"))
    test_files = sorted(glob.glob(f"{path}{base_name}*tst.dat"))

    # Iterate over each pair of training and testing files (folds)
    for train_file, test_file in zip(train_files, test_files):
        # Load training and testing data for the current fold
        X_train, y_train, X_test, y_test = load_data(train_file, test_file)
        
        # Calculate metrics (AUC and G-Mean) using the given model and data
        auc_score, g_mean_score = calculate_metrics(X_train, X_test, y_train, y_test, model_class, **kwargs)

        # Append scores to respective lists
        auc_scores.append(auc_score)
        g_mean_scores.append(g_mean_score)

    # Compute mean AUC and mean G-Mean across all folds
    mean_auc = np.mean(auc_scores)
    mean_g_mean = np.mean(g_mean_scores)

    # Return average metrics for the dataset
    return mean_auc, mean_g_mean
