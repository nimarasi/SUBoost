# Calculate Metrics
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def calculate_metrics(X_train, X_test, y_train, y_test, model_class, **kwargs):
    """calculate AUC and G-Mean."""
    
    # Initialize the model with given class and parameters
    model = model_class(**kwargs)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict probabilities if supported, otherwise predict labels
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else model.predict(X_test)
    
    # Calculate the Area Under the ROC Curve (AUC) score
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Predict class labels for the test set
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix components (True Negative, False Positive, False Negative, True Positive)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[1, -1]).ravel()
    
    # Calculate sensitivity (True Positive Rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate the G-Mean, the geometric mean of sensitivity and specificity
    g_mean = np.sqrt(sensitivity * specificity)

    # Return the AUC and G-Mean scores
    return auc_score, g_mean
