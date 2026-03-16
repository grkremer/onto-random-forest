import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Conversão DataFrame → {cluster_id: [feature_indices]}
# ─────────────────────────────────────────────────────────────

def _clusters_from_dataframe(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, List[int]]:
    """Converte um dataframe de features e clusters em um dicionário de mapeamento."""
    feat_col, cluster_col = df.columns[0], df.columns[1]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    clusters: Dict[str, List[int]] = {}
    for _, row in df.iterrows():
        name, cid = row[feat_col], str(row[cluster_col])
        if name in name_to_idx:
            clusters.setdefault(cid, []).append(name_to_idx[name])

    assigned = {i for idxs in clusters.values() for i in idxs}
    unassigned = [i for i, _ in enumerate(feature_names) if i not in assigned]
    if unassigned:
        clusters["__unassigned__"] = unassigned

    return clusters


# ─────────────────────────────────────────────────────────────
# Amostragem 50 / 50: cluster prioritário + fora dele
# ─────────────────────────────────────────────────────────────

def _sample_features_priority(
    clusters: Dict[str, List[int]],
    priority_key: str,
    m: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Sorteia m features com split 50% / 50%:
      - ceil(m/2) features vêm do cluster prioritário da árvore
      - floor(m/2) features vêm dos demais clusters combinados

    Se um lado tiver menos features do que sua cota, o deficit é absorvido pelo outro lado.
    """
    inside  = list(clusters[priority_key])
    outside = [f for k, feats in clusters.items()
               if k != priority_key for f in feats]

    quota_in  = (m + 1) // 2   # ceil(m/2)
    quota_out = m // 2          # floor(m/2)

    # Ajusta cotas se algum lado for menor que o necessário
    take_in  = min(quota_in,  len(inside))
    take_out = min(quota_out, len(outside))

    # Redistribui deficits
    take_in  = min(take_in  + (quota_out - take_out), len(inside))
    take_out = min(take_out + (quota_in  - take_in),  len(outside))

    selected = []
    if take_in:
        selected.extend(rng.choice(inside,  size=take_in,  replace=False).tolist())
    if take_out:
        selected.extend(rng.choice(outside, size=take_out, replace=False).tolist())

    return np.array(selected, dtype=int)


# ─────────────────────────────────────────────────────────────
# Árvore com Uniform Cluster Sampling (nó a nó)
# ─────────────────────────────────────────────────────────────

class _ClusterAwareTree:
    """
    Árvore de classificação interna. Reimplementa o loop de construção
    para controlar a amostragem de features nó a nó.
    """

    def __init__(self, clusters, m, n_classes, max_depth,
                 min_samples_split, min_samples_leaf, random_state):
        self.clusters = clusters
        self.m = m
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    @staticmethod
    def _gini(y):
        if not len(y):
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - float(np.dot(p, p))

    def _weighted_gini(self, yl, yr):
        n = len(yl) + len(yr)
        return len(yl) / n * self._gini(yl) + len(yr) / n * self._gini(yr)

    def _best_split(self, X, y, feat_ids):
        best_g, best_f, best_t = float("inf"), None, None
        for f in feat_ids:
            vals = X[:, f]
            uniq = np.unique(vals)
            if len(uniq) < 2:
                continue
            for thr in (uniq[:-1] + uniq[1:]) / 2.0:
                lm = vals <= thr
                rm = ~lm
                if lm.sum() < self.min_samples_leaf or rm.sum() < self.min_samples_leaf:
                    continue
                g = self._weighted_gini(y[lm], y[rm])
                if g < best_g:
                    best_g, best_f, best_t = g, f, thr
        return best_f, best_t

    def _build(self, X, y, depth, priority_key, rng):
        # Critérios de parada
        if (len(np.unique(y)) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return self._leaf(y)

        # Raiz (priority_key ainda desconhecido): sorteia uniformemente
        if priority_key is None:
            all_feats = list(range(X.shape[1]))
            feat_ids  = np.array(rng.choice(all_feats, size=self.m, replace=False))
        else:
            feat_ids  = _sample_features_priority(self.clusters, priority_key, self.m, rng)

        f, thr = self._best_split(X, y, feat_ids)

        if f is None:
            return self._leaf(y)

        # Feature vencedora define o cluster prioritário de toda a árvore
        if priority_key is None:
            priority_key = self._feature_to_cluster[f]
            self.priority_cluster_ = priority_key

        lm = X[:, f] <= thr
        return {
            "f": f, "thr": thr,
            "left":  self._build(X[lm],  y[lm],  depth + 1, priority_key, rng),
            "right": self._build(X[~lm], y[~lm], depth + 1, priority_key, rng),
        }

    def _leaf(self, y):
        p = np.zeros(self.n_classes)
        if len(y):
            p = np.bincount(y, minlength=self.n_classes) / len(y)
        return {"proba": p}

    def fit(self, X, y):
        self._feature_to_cluster = {
            feat: key
            for key, feats in self.clusters.items()
            for feat in feats
        }
        self.priority_cluster_ = None
        self._root = self._build(X, y, depth=0, priority_key=None,
                                 rng=check_random_state(self.random_state))
        return self

    def _walk(self, x, node):
        if "proba" in node:
            return node["proba"]
        return self._walk(x, node["left"] if x[node["f"]] <= node["thr"] else node["right"])

    def predict_proba(self, X):
        return np.array([self._walk(x, self._root) for x in X])


# ─────────────────────────────────────────────────────────────
# ClusterAwareRandomForestClassifier
# ─────────────────────────────────────────────────────────────

class ClusterAwareRandomForestClassifier(RandomForestClassifier):
    """
    Random Forest Classifier guiado por ontologias (Onto-RF).

    A raiz de cada árvore sorteia features uniformemente (RF padrão).
    O cluster da feature vencedora no primeiro split torna-se o cluster prioritário 
    de toda a árvore: os próximos splits usam 50% de features desse cluster 
    e 50% dos demais.
    """

    def __init__(
        self,
        clusters: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=1,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.clusters = clusters
        self.feature_names = feature_names

    def _resolve_m(self, n_features: int) -> int:
        mf = self.max_features
        if mf == "sqrt":  return max(1, int(np.sqrt(n_features)))
        if mf == "log2":  return max(1, int(np.log2(n_features)))
        if isinstance(mf, float): return max(1, int(mf * n_features))
        if isinstance(mf, int):   return mf
        if mf is None: return n_features
        raise ValueError(f"max_features inválido: {mf!r}")

    def _fit_single_tree(self, X, y, seed):
        rng = check_random_state(seed)
        if self.bootstrap:
            idx = rng.choice(len(y), size=len(y), replace=True)
            X, y = X[idx], y[idx]

        return _ClusterAwareTree(
            clusters=self.clusters_,
            m=self.m_,
            n_classes=self.n_classes_,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=rng.randint(0, 2**31),
        ).fit(X, y)

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            feature_names_ = list(X.columns)
            X = X.values
        elif self.feature_names is not None:
            feature_names_ = list(self.feature_names)
        else:
            raise ValueError("Informe feature_names ou passe X como pd.DataFrame.")

        self.clusters_ = _clusters_from_dataframe(self.clusters, feature_names_)
        self.feature_names_in_ = np.array(feature_names_)

        self.classes_       = np.unique(y)
        self.n_classes_     = len(self.classes_)
        self.n_outputs_     = 1
        self.n_features_in_ = X.shape[1]
        self.m_             = self._resolve_m(X.shape[1])

        y_enc = np.searchsorted(self.classes_, y)
        rng   = check_random_state(self.random_state)
        seeds = rng.randint(0, 2**31, size=self.n_estimators)

        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single_tree)(X, y_enc, int(s)) for s in seeds
        )

        self.tree_priority_clusters_ = [t.priority_cluster_ for t in self.estimators_]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
