import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier as DecisionTree

# -------------------------
# Optimized Tree implementation
# -------------------------


# -------------------------
# Optimized Random Forest Implementation
# -------------------------
class MyRandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                 verbose=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.trees = []
        self.n_classes_ = None
        self.n_features_ = None
        self.oob_score_ = None
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def _get_max_features(self, n_features):
        """Determine the number of features to consider for each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features)) + 1
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features is None:
            return n_features
        else:
            return self.max_features

    def _build_tree(self, X, y, tree_idx):
        """Build a single tree with optional bootstrap sampling"""
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Use bootstrap sampling
            idxs = np.random.randint(0, n_samples, n_samples)
            X_sample, y_sample = X[idxs], y[idxs]
            oob_idxs = ~np.isin(np.arange(n_samples), np.unique(idxs))
        else:
            # Use all samples
            X_sample, y_sample = X, y
            oob_idxs = np.zeros(n_samples, dtype=bool)
        
        max_features = self._get_max_features(X.shape[1])
        
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state + tree_idx if self.random_state else None
        )
        tree.fit(X_sample, y_sample)
        
        return tree, oob_idxs

    def fit(self, X, y):
        """Fit the random forest classifier"""
        X = self._validate_data(X)
        y = np.asarray(y)
        
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        if self.verbose:
            print(f"Training Random Forest with {self.n_trees} trees...")
        
        # Build trees in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._build_tree)(X, y, i) for i in range(self.n_trees)
        )
        
        self.trees = [result[0] for result in results]
        
        # Compute OOB score if requested
        if self.oob_score and self.bootstrap:
            if self.verbose:
                print("Computing OOB score...")
            self._compute_oob_score(X, y, results)
        
        # Compute feature importances
        #self._compute_feature_importances()
        
        return self

    def _compute_oob_score(self, X, y, tree_results):
        """Compute out-of-bag score"""
        n_samples = X.shape[0]
        oob_predictions = [None] * n_samples
        oob_counts = np.zeros(n_samples, dtype=int)
        
        # Collect OOB predictions
        for tree, oob_idxs in tree_results:
            oob_mask = oob_idxs
            if np.any(oob_mask):
                oob_X = X[oob_mask]
                preds = tree.predict(oob_X)
                
                oob_indices = np.where(oob_mask)[0]
                for idx, pred in zip(oob_indices, preds):
                    if oob_predictions[idx] is None:
                        oob_predictions[idx] = []
                    oob_predictions[idx].append(pred)
                    oob_counts[idx] += 1
        
        # Calculate OOB score
        oob_mask = oob_counts > 0
        if np.any(oob_mask):
            y_oob = y[oob_mask]
            preds_oob = []
            
            for idx in np.where(oob_mask)[0]:
                pred = Counter(oob_predictions[idx]).most_common(1)[0][0]
                preds_oob.append(pred)
            
            self.oob_score_ = np.mean(y_oob == preds_oob)
            
            if self.verbose:
                print(f"OOB Score: {self.oob_score_:.4f}")
        else:
            self.oob_score_ = 0.0
            if self.verbose:
                print("No OOB samples available for scoring")

    def _compute_feature_importances(self):
        """Compute feature importances based on mean decrease in impurity"""
        importances = np.zeros(self.n_features_)
        
        for tree in self.trees:
            self._accumulate_tree_importance(tree.tree, importances)
        
        # Normalize importances
        if np.sum(importances) > 0:
            importances /= np.sum(importances)
        
        self.feature_importances_ = importances

    def _accumulate_tree_importance(self, node, importances, parent_impurity=1.0):
        """Recursively accumulate feature importance from tree nodes"""
        if not isinstance(node, tuple):
            return
        
        if len(node) == 4:  # Internal node
            feat_idx, threshold, left, right = node
            
            # Calculate impurity decrease (simplified)
            if isinstance(left, tuple) and len(left) == 2:
                left_impurity = 1.0 - np.max(left[1] / np.sum(left[1])) if np.sum(left[1]) > 0 else 0
            else:
                left_impurity = 0
                
            if isinstance(right, tuple) and len(right) == 2:
                right_impurity = 1.0 - np.max(right[1] / np.sum(right[1])) if np.sum(right[1]) > 0 else 0
            else:
                right_impurity = 0
            
            impurity_decrease = parent_impurity - (left_impurity + right_impurity) / 2
            importances[feat_idx] += max(0, impurity_decrease)
            
            # Recurse children
            self._accumulate_tree_importance(left, importances, left_impurity)
            self._accumulate_tree_importance(right, importances, right_impurity)

    def predict(self, X):
        """Predict class labels for samples in X"""
        X = self._validate_data(X, predict=True)
        
        if not self.trees:
            raise ValueError("The forest has not been fitted yet.")
        
        # Parallel prediction
        if self.n_jobs != 1:
            tree_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict)(X) for tree in self.trees
            )
            tree_preds = np.array(tree_preds)
        else:
            tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority voting
        return self._majority_vote(tree_preds)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        X = self._validate_data(X, predict=True)
        
        if not self.trees:
            raise ValueError("The forest has not been fitted yet.")
        
        # Get probabilities from all trees
        if self.n_jobs != 1:
            tree_probas = Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict_proba)(X) for tree in self.trees
            )
        else:
            tree_probas = [tree.predict_proba(X) for tree in self.trees]
        
        # Average probabilities
        avg_proba = np.mean(tree_probas, axis=0)
        return avg_proba

    def _majority_vote(self, tree_preds):
        """Perform majority voting on tree predictions"""
        # tree_preds shape: (n_trees, n_samples)
        n_samples = tree_preds.shape[1]
        final_predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            counts = np.bincount(tree_preds[:, i], minlength=self.n_classes_)
            final_predictions[i] = np.argmax(counts)
        
        return final_predictions

    def _validate_data(self, X, predict=False):
        """Validate and convert input data"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if predict and X.shape[1] != self.n_features_:
            raise ValueError(f"Number of features ({X.shape[1]}) does not match fitted data ({self.n_features_})")
        
        return X

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        """Set the parameters of this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self