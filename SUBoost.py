# Models
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans


# Helper function: approximate C4.5 decision tree using scikit-learn's DecisionTreeClassifier with entropy criterion
def c45_tree(depth):
    # Return a decision tree classifier with entropy criterion and given max depth
    return DecisionTreeClassifier(criterion='entropy', max_depth=depth)


# Base class for all boosting algorithms
class BaseBoost:
    def __init__(self, n_estimators=100, depth=5):
        """
        Initializes the base boosting class.
        :param n_estimators: Number of weak learners to train
        :param depth: Max depth of each decision tree
        """
        self.n_estimators = n_estimators
        self.depth = depth
        self.estimators_ = []         # List to store trained estimators
        self.estimator_alphas_ = []   # List to store weights (alphas) for each estimator

    def _initialize(self, X, y, sample_weight=None):
        """
        Prepare data and initialization for boosting.
        Ensures X and y are validated and sets initial sample weights.
        """
        X, y = check_X_y(X, y)  # Validate shapes and types of X and y
        self.classes_ = np.unique(y)  # Extract unique class labels
        self.n_classes_ = len(self.classes_)
        if sample_weight is None:
            # Initialize uniform weights if not provided
            sample_weight = np.full(len(y), 1 / len(y))
        return X, y, sample_weight

    def _aggregate_predictions(self, X):
        """
        Aggregate predictions from all weak learners using their respective alphas.
        Returns signed sum of predictions indicating class label.
        """
        check_is_fitted(self)  # Check model was fitted before predicting
        X = check_array(X)
        predictions = np.array([est.predict(X) for est in self.estimators_])  # Predictions of all estimators
        weighted_sum = np.dot(self.estimator_alphas_, predictions)  # Weighted sum of predictions
        return np.sign(weighted_sum)  # Return sign as aggregated prediction (+1 or -1)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Converts aggregated prediction signs to original class labels.
        """
        final_pred = self._aggregate_predictions(X)
        # Map -1 back to first class, +1 to second class
        return np.where(final_pred == -1, self.classes_[0], self.classes_[1])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        Uses logistic function over the signed sum to estimate probabilities.
        """
        final_pred = self._aggregate_predictions(X)
        return 1 / (1 + np.exp(-final_pred))


# -------------------------- AdaBoost --------------------------
class AdaBoost(BaseBoost):
    def fit(self, X, y):
        """
        Fit AdaBoost classifier using decision stumps (C4.5 approximation).
        """
        X, y, sample_weight = self._initialize(X, y)
        # Transform labels to {-1, +1}
        y_transformed = np.where(y == self.classes_[0], -1, 1)

        for _ in range(self.n_estimators):
            clf = c45_tree(self.depth)                   # Create a new decision tree
            clf.fit(X, y_transformed, sample_weight=sample_weight)  # Fit weighted classifier
            y_pred = clf.predict(X)                       # Predict on training data

            incorrect = (y_pred != y_transformed)        # Identify misclassified samples
            error = np.sum(sample_weight[incorrect]) / np.sum(sample_weight)  # Compute weighted error
            if error == 0: error = 1e-10                  # Avoid division by zero
            if error >= 0.5: break                        # Stop if error too large

            # Compute estimator weight (alpha)
            alpha = 0.5 * np.log((1.0 - error) / error)
            # Update example weights: increase weights for misclassified samples
            sample_weight *= np.exp(-alpha * y_transformed * y_pred)
            # Normalize weights
            sample_weight /= np.sum(sample_weight)

            # Store the estimator and its weight
            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)
        return self


# -------------------------- AdaC1 --------------------------
class AdaC1(BaseBoost):
    def __init__(self, n_estimators=100, depth=10, cost_misclassifying_minority=2.0):
        """
        AdaC1 boosting variant initializes with cost parameter for minority class misclassification.
        """
        super().__init__(n_estimators, depth)
        self.cost_misclassifying_minority = cost_misclassifying_minority

    def fit(self, X, y):
        """
        Fit AdaC1 boosting classifier where cost of misclassifying minority class is higher.
        """
        X, y, sample_weight = self._initialize(X, y)
        counts = Counter(y)                          # Count samples per class
        minority_class = min(counts, key=counts.get) # Identify minority class
        # Transform labels: minority as 1, others as -1
        y_transformed = np.where(y == minority_class, 1, -1)
        # Set cost vector for weighting errors differently
        cost_vector = np.where(y_transformed == 1, self.cost_misclassifying_minority, 1.0)

        for _ in range(self.n_estimators):
            clf = c45_tree(self.depth)
            clf.fit(X, y_transformed, sample_weight=sample_weight)
            y_pred = clf.predict(X)

            incorrect = (y_pred != y_transformed)
            # Compute weighted error with cost vector
            error = np.sum(sample_weight[incorrect] * cost_vector[incorrect]) / np.sum(sample_weight * cost_vector)
            if error == 0: error = 1e-10
            if error >= 0.5: break

            alpha = 0.5 * np.log((1.0 - error) / error)
            # Update sample weights with cost
            sample_weight *= np.exp(-alpha * y_transformed * y_pred * cost_vector)
            sample_weight /= np.sum(sample_weight)

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)
        return self


# -------------------------- AdaCost --------------------------
class AdaCost(BaseBoost):
    def __init__(self, n_estimators=100, depth=10, cost_misclassifying_minority=2.0):
        """
        AdaCost boosting variant with cost-sensitive adjustments.
        """
        super().__init__(n_estimators, depth)
        self.cost_misclassifying_minority = cost_misclassifying_minority

    def fit(self, X, y):
        """
        Fit AdaCost boosting classifier by adjusting sample weights after prediction.
        """
        X, y, sample_weight = self._initialize(X, y)
        counts = Counter(y)
        minority_class = min(counts, key=counts.get)
        y_transformed = np.where(y == minority_class, 1, -1)
        cost_vector = np.where(y_transformed == 1, self.cost_misclassifying_minority, 1.0)

        for _ in range(self.n_estimators):
            clf = c45_tree(self.depth)
            clf.fit(X, y_transformed, sample_weight=sample_weight)
            y_pred = clf.predict(X)

            incorrect = (y_pred != y_transformed)
            error = np.sum(sample_weight[incorrect]) / np.sum(sample_weight)
            if error == 0: error = 1e-10
            if error >= 0.5: break

            alpha = 0.5 * np.log((1.0 - error) / error)
            # Update weights with cost factor applied only on misclassified samples
            sample_weight *= np.exp(-alpha * y_transformed * y_pred) * np.where(incorrect, cost_vector, 1.0)
            sample_weight /= np.sum(sample_weight)

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)
        return self


# -------------------------- SUBoost --------------------------
class SUBoost(BaseBoost):
    def __init__(self, n_estimators=100, depth=5):
        """
        SUBoost boosting variant that balances samples via undersampling majority class on the fly.
        """
        super().__init__(n_estimators, depth)

    def fit(self, X, y):
        """
        Fit SUBoost classifier using iterative undersampling of the majority class during boosting.
        """
        X, y, w = self._initialize(X, y)
        counts = Counter(y)
        major_class = max(counts, key=counts.get)
        minor_class = min(counts, key=counts.get)
        n_major, n_minor = counts[major_class], counts[minor_class]

        # Initialize sample weights: minor and major classes weighted inversely proportional to their counts
        w = np.where(y == minor_class, 1/n_minor, 1/n_major)

        for i in range(self.n_estimators):
            clf = c45_tree(self.depth)

            if i != 0:
                # Find indices of major class samples predicted correctly
                major_correct_indices = np.where((y == major_class) & (y == predictions))[0]
                major_correct = len(major_correct_indices)

                # Remove majority samples to balance with minority according to prediction correctness
                if n_major - major_correct < n_minor:
                    to_delete = n_major - n_minor
                    delete_indices = np.random.choice(major_correct_indices, to_delete, replace=False)
                    mask = np.ones(len(y), dtype=bool)
                    mask[delete_indices] = False
                    X_resampled, y_resampled, w_resampled = X[mask], y[mask], w[mask]

                elif n_major - major_correct == n_minor:
                    mask = np.ones(len(y), dtype=bool)
                    mask[major_correct_indices] = False
                    X_resampled, y_resampled, w_resampled = X[mask], y[mask], w[mask]

                elif n_major - major_correct > n_minor:
                    major_misclassified_indices = np.where((y == major_class) & (y != predictions))[0]
                    to_delete = n_major - major_correct - n_minor
                    delete_indices = np.random.choice(major_misclassified_indices, to_delete, replace=False)
                    mask = np.ones(len(y), dtype=bool)
                    mask[major_correct_indices] = False
                    mask[delete_indices] = False
                    X_resampled, y_resampled, w_resampled = X[mask], y[mask], w[mask]

            if i == 0:
                clf.fit(X, y, sample_weight=w)
            else:
                # Train on resampled data with balanced majority and minority classes
                clf.fit(X_resampled, y_resampled, sample_weight=w_resampled)

            predictions = clf.predict(X)
            error = np.sum(w[y != predictions])
            EPS = 1e-10
            alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))

            # Update weights exponentially based on misclassified samples
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)

        return self


# -------------------------- RUSBoost --------------------------
class RUSBoost(BaseBoost):
    def __init__(self, n_estimators=10, depth=5):
        """
        RUSBoost employs random undersampling of majority class before training each weak learner.
        """
        super().__init__(n_estimators, depth)

    def fit(self, X, y):
        """
        Fit RUSBoost classifier by undersampling the majority class in each boosting round.
        """
        X, y, w = self._initialize(X, y)
        counts = Counter(y)
        major_class = max(counts, key=counts.get)
        minor_class = min(counts, key=counts.get)
        n_major, n_minor = counts[major_class], counts[minor_class]

        # Initialize weights inversely proportional to class counts
        w = np.where(y == minor_class, 1/n_minor, 1/n_major)

        for i in range(self.n_estimators):
            clf = c45_tree(self.depth)

            if i != 0:
                # Randomly remove samples from majority class to balance classes
                major_indices = np.where(y == major_class)[0]
                to_delete = n_major - n_minor
                delete_indices = np.random.choice(major_indices, to_delete, replace=False)
                mask = np.ones(len(y), dtype=bool)
                mask[delete_indices] = False
                X_resampled, y_resampled, w_resampled = X[mask], y[mask], w[mask]

            if i == 0:
                clf.fit(X, y, sample_weight=w)
            else:
                # Fit on resampled balanced data
                clf.fit(X_resampled, y_resampled, sample_weight=w_resampled)

            predictions = clf.predict(X)
            error = np.sum(w[y != predictions])
            EPS = 1e-10
            alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))

            # Update sample weights for next iteration
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)

        return self


# -------------------------- SMOTEBoost --------------------------
class SMOTEBoost(BaseBoost):
    def __init__(self, n_estimators=100, depth=10, k_neighbors=3):
        """
        SMOTEBoost performs SMOTE oversampling before each weak learner after the first.
        """
        super().__init__(n_estimators, depth)
        self.k_neighbors = k_neighbors

    def fit(self, X, y):
        """
        Fit SMOTEBoost classifier with SMOTE oversampling used after the first iteration.
        """
        X, y, w = self._initialize(X, y)
        smote = SMOTE(k_neighbors=self.k_neighbors)
        for i in range(self.n_estimators):
            if i != 0:
                # Apply SMOTE to create synthetic minority samples to balance classes
                X_resampled, y_resampled = smote.fit_resample(X, y)
                w_resampled = np.full(len(y_resampled), 1 / len(y_resampled))  # Equal weights after resampling
                clf = c45_tree(self.depth)
                clf.fit(X_resampled, y_resampled, sample_weight=w_resampled)
            else:
                clf = c45_tree(self.depth)
                clf.fit(X, y, sample_weight=w)

            predictions = clf.predict(X)
            error = np.sum(w[y != predictions])
            EPS = 1e-10
            alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))
            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)
        return self


# -------------------------- OUBoost --------------------------
# Implementation of the Peak Undersampling Algorithm used in OUBoost variant

class PeakUndersampler:
    """
    Cluster-based undersampling algorithm "Peak" from OUBoost paper.
    Uses k-means clustering to select representative majority samples probabilistically.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y, w, target_size):
        """
        Perform undersampling using clustering until the number of samples equals target_size.
        """
        n_samples = X.shape[0]
        if target_size >= n_samples:
            # No undersampling needed if target size greater than current samples
            return X, y, w

        k = int(np.sqrt(n_samples))  # Number of clusters
        if k <= 1:
            # If too few samples, randomly sample without replacement
            indices = np.random.choice(n_samples, size=target_size, replace=False)
            return X[indices], y[indices], w[indices]

        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_assignments = kmeans.fit_predict(X)  # Cluster assignments for each sample
        cluster_labels, counts = np.unique(cluster_assignments, return_counts=True)
        gamma = counts / n_samples  # Proportion of samples in each cluster
        # Inverse proportional selection probability for each cluster
        selection_prob_cluster = 1 / (gamma + 1e-8)

        prob_map = dict(zip(cluster_labels, selection_prob_cluster))  # Map cluster to selection probs
        sample_probabilities = np.array([prob_map[c] for c in cluster_assignments])
        sample_probabilities /= np.sum(sample_probabilities)  # Normalize probabilities

        # Randomly select samples according to computed probabilities
        chosen_indices = np.random.choice(
            np.arange(n_samples), size=target_size, replace=False, p=sample_probabilities
        )
        return X[chosen_indices], y[chosen_indices], w[chosen_indices]


class OUBoost(BaseBoost):
    def __init__(self, n_estimators=10, depth=5, random_state=None):
        """
        OUBoost integrates Peak Undersampler and SMOTE for imbalanced learning.
        """
        super().__init__(n_estimators, depth)
        self.random_state = random_state
        self.peak_sampler_ = PeakUndersampler(random_state=self.random_state)
        self.smote_ = SMOTE(random_state=self.random_state)

    def fit(self, X, y):
        """
        Fit OUBoost classifier using peak undersampling and SMOTE oversampling in boosting rounds.
        """
        # Initialize base class and weights
        X, y, w = super()._initialize(X, y)

        # Convert original labels to {-1, +1} for computations
        y_transformed = np.where(y == self.classes_[0], -1, 1)

        counts = Counter(y_transformed)
        major_label, n_maj = counts.most_common(1)[0]
        minor_label, n_min = counts.most_common(2)[-1]

        for i in range(self.n_estimators):
            clf = c45_tree(self.depth)

            # Separate majority and minority class indices
            maj_idx = np.where(y_transformed == major_label)[0]
            min_idx = np.where(y_transformed == minor_label)[0]

            # Undersample majority class with Peak algorithm to size of minority class
            X_maj, y_maj_t, w_maj = X[maj_idx], y_transformed[maj_idx], w[maj_idx]
            X_maj_res, y_maj_res_t, _ = self.peak_sampler_.fit_resample(
                X_maj, y_maj_t, w_maj, target_size=n_min
            )

            # Combine resampled majority with minority samples
            X_min, y_min_t = X[min_idx], y_transformed[min_idx]
            X_temp = np.vstack([X_maj_res, X_min])
            y_temp_t = np.hstack([y_maj_res_t, y_min_t])

            # Oversample combined data with SMOTE to balance further
            X_resampled, y_resampled_t = self.smote_.fit_resample(X_temp, y_temp_t)

            # Train weak learner on resampled balanced dataset
            clf.fit(X_resampled, y_resampled_t)

            preds = clf.predict(X)  # Predict on original samples

            misclassified = (preds != y_transformed)
            error = np.sum(w[misclassified])

            EPS = 1e-10
            error = np.clip(error, EPS, 1 - EPS)
            alpha = 0.5 * np.log((1 - error) / error)

            # Update sample weights for next iteration
            w *= np.exp(-alpha * y_transformed * preds)
            w /= np.sum(w)

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha)

        return self


# -------------------------- IMBoost --------------------------
class IMBoost(BaseBoost):
    def __init__(self, n_estimators=10, depth=5):
        """
        IMBoost applies different error computations and weight updates for minority and majority classes.
        """
        super().__init__(n_estimators=n_estimators, depth=depth)

    def fit(self, X, y):
        """
        Fit IMBoost classifier with class-specific error calculations and resampling.
        """
        # Initialize; labels expected to be +1 for minority, -1 for majority
        X, y, _ = self._initialize(X, y)

        n_samples = X.shape[0]
        X_t, y_t = X.copy(), y.copy()

        # Initialize sample weights inversely proportional to class sizes
        counts = Counter(y_t)
        N_min = counts.get(1, 0)
        N_maj = counts.get(-1, 0)
        D_t = np.ones(n_samples)
        if N_min > 0:
            D_t[y_t == 1] = 1 / N_min
        if N_maj > 0:
            D_t[y_t == -1] = 1 / N_maj

        for _ in range(self.n_estimators):
            clf = c45_tree(self.depth)
            clf.fit(X_t, y_t, sample_weight=D_t)
            y_pred = clf.predict(X_t)

            minority_mask = (y_t == 1)
            majority_mask = (y_t == -1)

            # Calculate weighted error separately for minority and majority
            incorrect_min = (y_pred[minority_mask] != y_t[minority_mask])
            error_min = np.sum(D_t[minority_mask][incorrect_min])

            incorrect_maj = (y_pred[majority_mask] != y_t[majority_mask])
            error_maj = np.sum(D_t[majority_mask][incorrect_maj])

            EPS = 1e-10
            error_min = np.clip(error_min, EPS, 1.0 - EPS)
            error_maj = np.clip(error_maj, EPS, 1.0 - EPS)

            # Compute classifier weight alpha combining minority and majority errors
            alpha_min = 0.5 * np.log((1.0 - error_min) / error_min)
            alpha_maj = 0.5 * np.log((1.0 - error_maj) / error_maj)
            alpha_t = alpha_min + alpha_maj

            self.estimators_.append(clf)
            self.estimator_alphas_.append(alpha_t)

            # Update sample weights for minority and majority differently
            D_updated = D_t.copy()
            D_updated[minority_mask] *= np.exp(-alpha_min * y_t[minority_mask] * y_pred[minority_mask])
            D_updated[majority_mask] *= np.exp(-alpha_maj * y_t[majority_mask] * y_pred[majority_mask])

            Z_t = np.sum(D_updated)
            if Z_t <= 0: break
            D_for_resampling = D_updated / Z_t

            # Resample training data based on updated weights for next boosting round
            current_n_samples = X_t.shape[0]
            resample_indices = np.random.choice(
                np.arange(current_n_samples),
                size=n_samples,
                replace=True,
                p=D_for_resampling
            )
            X_t = X_t[resample_indices]
            y_t = y_t[resample_indices]

            # Reinitialize weights after resampling based on new distribution
            counts_next = Counter(y_t)
            N_min_next = counts_next.get(1, 0)
            N_maj_next = counts_next.get(-1, 0)

            if N_min_next == 0 or N_maj_next == 0: break

            D_t = np.ones(n_samples)
            if N_min_next > 0:
                D_t[y_t == 1] = 1 / N_min_next
            if N_maj_next > 0:
                D_t[y_t == -1] = 1 / N_maj_next

        return self
