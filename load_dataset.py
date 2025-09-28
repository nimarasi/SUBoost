# Load Dataset
import pandas as pd
import numpy as np

def load_data(train_file_path, test_file_path):
    """
    Loads and preprocesses training and testing data separately to prevent
    data leakage. All preprocessing steps (imputation, encoding) are learned
    from the training data and then applied to both datasets.
    """
    # --- 1. Read Data without Headers ---
    # Find the line where the actual data starts by skipping metadata lines 
    # that begin with '@' (typical in ARFF files)
    start_line = 0
    with open(train_file_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip().startswith('@'):
                start_line = i
                break

    # Load the CSV data into pandas DataFrames without headers,
    # skipping the initial metadata lines
    train_df = pd.read_csv(train_file_path, header=None, skiprows=start_line, sep=',')
    test_df = pd.read_csv(test_file_path, header=None, skiprows=start_line, sep=',')

    # --- 2. Initial Cleaning (Applied Separately to training and testing data) ---
    for df in [train_df, test_df]:
        # Remove trailing and leading whitespace in columns of string type
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()
        # Replace placeholder '?' with NaN for correct missing value handling
        df.replace('?', np.nan, inplace=True)

    # --- 3. Separate Features (X) and Target (y) ---
    # The target column is assumed to be the last column in the dataset
    target_col_index = train_df.shape[1] - 1
    X_train = train_df.drop(columns=[target_col_index])
    y_train_raw = train_df[target_col_index]

    X_test = test_df.drop(columns=[target_col_index])
    y_test_raw = test_df[target_col_index]

    # Store the original feature columns order
    original_cols_order = X_train.columns.tolist()

    # --- 4. Learn Imputation Strategies from Training Data ONLY ---
    imputation_values = {}
    categorical_cols = []

    for col in original_cols_order:
        # Try converting the column to numeric values, coercing errors to NaN
        converted_col = pd.to_numeric(X_train[col], errors='coerce')

        # If conversion increases NaNs, it indicates the column is categorical
        if X_train[col].isnull().sum() < converted_col.isnull().sum():
            categorical_cols.append(col)
            # Use mode (most frequent value) as imputation for categorical columns
            imputation_values[col] = X_train[col].mode()[0]
        else:
            # For numeric columns, use median as imputation value
            imputation_values[col] = X_train[col].median()
            # Convert columns to numeric type in both train and test sets
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # --- 5. Apply Imputation to Fill Missing Values ---
    # Fill missing values in both training and testing features using learned values
    X_train.fillna(imputation_values, inplace=True)
    X_test.fillna(imputation_values, inplace=True)

    # --- 6. One-Hot Encoding for Categorical Features ---
    # Learn one-hot encoding from training data and apply it to both datasets
    # Drop the first category to avoid dummy variable trap, and use integer dtype
    X_train_processed = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
    X_test_processed = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)

    # Align columns between train and test datasets to ensure consistency
    train_cols = X_train_processed.columns
    test_cols = X_test_processed.columns

    # Add any missing columns in the test dataset with zeros
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_processed[c] = 0

    # Remove any columns present in test but not in train dataset
    extra_in_test = set(test_cols) - set(train_cols)
    X_test_processed.drop(columns=list(extra_in_test), inplace=True)

    # Reorder test columns to match training columns exactly
    X_test_processed = X_test_processed[train_cols]

    # --- 7. Finalize Datasets ---
    # Map target labels to numeric values: 'negative'->-1, 'positive'->1
    y_train = y_train_raw.map({'negative': -1, 'positive': 1}).rename('output')
    y_test = y_test_raw.map({'negative': -1, 'positive': 1}).rename('output')

    # Rename feature columns to a generic format dim1, dim2, ..., for anonymity or clarity
    num_features = X_train_processed.shape[1]
    new_column_names = [f'dim{i+1}' for i in range(num_features)]
    X_train_processed.columns = new_column_names
    X_test_processed.columns = new_column_names

    # Return preprocessed train/test features and targets ready for modeling
    return X_train_processed, y_train, X_test_processed, y_test
