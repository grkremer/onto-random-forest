import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de Similaridade de Cosseno
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity_vec(anchor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calcula a similaridade de cosseno entre `anchor` (shape D,) e cada
    linha de `matrix` (shape N×D). Retorna vetor de tamanho N.
    """
    anchor_norm = anchor / (np.linalg.norm(anchor) + 1e-12)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix.dot(anchor_norm) / norms.squeeze()


def _similarity_weights(
    anchor_idx: int,
    emb_matrix: np.ndarray,   # (n_features, D) – NaN rows para genes sem embedding
    rng: np.random.RandomState,
    temperature: float = 0.5,
) -> np.ndarray:
    """
    Retorna um vetor de probabilidades de amostragem para cada feature.

    - Features sem embedding recebem peso igual à média das demais após softmax.
    - `temperature` controla o quão 'peaked' é a distribuição:
        valores menores → features mais similares dominam ainda mais.
    """
    anchor_emb = emb_matrix[anchor_idx]

    # Se a feature âncora não possui embedding válido, retorna prob uniforme
    if np.isnan(anchor_emb[0]):
        return np.ones(len(emb_matrix)) / len(emb_matrix)

    # Determina quais features têm embedding válido
    has_emb = ~np.isnan(emb_matrix[:, 0])

    sims = np.zeros(len(emb_matrix))
    if has_emb.sum() > 1:
        sims[has_emb] = _cosine_similarity_vec(anchor_emb, emb_matrix[has_emb])

    # Converete similaridades em probabilidades via softmax com temperatura
    logits = sims / temperature
    # Estabiliza numericamente
    logits -= logits[has_emb].max() if has_emb.any() else 0.0
    exp_l = np.exp(logits)
    # Features sem embedding recebem a probabilidade média do grupo válido
    mean_w = exp_l[has_emb].mean() if has_emb.any() else 1.0
    exp_l[~has_emb] = mean_w

    probs = exp_l / exp_l.sum()
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Árvore guiada por Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class _EmbeddingGuidedTree:
    """
    Árvore de decisão interna do EmbeddingGuidedRandomForestClassifier.

    Lógica de amostragem de features:
      - Nó raiz: amostragem **uniforme** (comportamento RF padrão).
      - Demais nós: amostragem **proporcional à similaridade de cosseno**
        com o embedding da feature vencedora do split imediatamente pai.
    """

    def __init__(self, emb_matrix, m, n_classes,
                 max_depth, min_samples_split, min_samples_leaf,
                 random_state, temperature):
        self.emb_matrix = emb_matrix          # (n_features, D)
        self.m = m
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.temperature = temperature

    # ── Métricas de impureza ──────────────────────────────────────────────────

    @staticmethod
    def _gini(y):
        if not len(y):
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - float(np.dot(p, p))

    def _weighted_gini(self, yl, yr):
        n = len(yl) + len(yr)
        return len(yl) / n * self._gini(yl) + len(yr) / n * self._gini(yr)

    # ── Melhor split dado um subconjunto de features ──────────────────────────

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

    # ── Construção recursiva ──────────────────────────────────────────────────

    def _build(self, X, y, depth, anchor_feat_idx, rng):
        n_feats = X.shape[1]

        # Critérios de parada
        if (len(np.unique(y)) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return self._leaf(y)

        # Seleção de features candidatas
        if anchor_feat_idx is None:
            # Nó raiz: uniforme
            feat_ids = rng.choice(n_feats, size=self.m, replace=False)
        else:
            # Demais nós: ponderado por similaridade de cosseno
            probs = _similarity_weights(
                anchor_feat_idx, self.emb_matrix, rng, self.temperature
            )
            feat_ids = rng.choice(n_feats, size=self.m, replace=False, p=probs)

        f, thr = self._best_split(X, y, feat_ids)

        if f is None:
            return self._leaf(y)

        lm = X[:, f] <= thr
        return {
            "f": f, "thr": thr,
            "left":  self._build(X[lm],  y[lm],  depth + 1, f, rng),
            "right": self._build(X[~lm], y[~lm], depth + 1, f, rng),
        }

    def _leaf(self, y):
        p = np.zeros(self.n_classes)
        if len(y):
            p = np.bincount(y, minlength=self.n_classes) / len(y)
        return {"proba": p}

    # ── Treino e Inferência ───────────────────────────────────────────────────

    def fit(self, X, y):
        self._root = self._build(
            X, y, depth=0, anchor_feat_idx=None,
            rng=check_random_state(self.random_state)
        )
        return self

    def _walk(self, x, node):
        if "proba" in node:
            return node["proba"]
        return self._walk(x, node["left"] if x[node["f"]] <= node["thr"] else node["right"])

    def predict_proba(self, X):
        return np.array([self._walk(x, self._root) for x in X])


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingGuidedRandomForestClassifier  (classe principal)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingGuidedRandomForestClassifier(RandomForestClassifier):
    """
    Random Forest guiado por embeddings de ontologias (Onto-RF v2).

    O primeiro split de cada árvore seleciona features **uniformemente**,
    como o RF padrão. A partir do segundo nó, as features são amostradas
    com **probabilidade proporcional à similaridade de cosseno** com o
    embedding da feature vencedora do split imediatamente anterior.

    Parâmetros
    ----------
    embeddings : pd.DataFrame
        DataFrame cujo índice são nomes de genes e as colunas são as
        dimensões do embedding (ex: saída do OWL2Vec* / mOWL).
    temperature : float, default=0.5
        Temperatura do softmax aplicado às similaridades. Valores menores
        tornam a distribuição mais concentrada nas features mais próximas.
    n_estimators, max_features, max_depth, ... : igual ao sklearn RF.
    """

    def __init__(
        self,
        embeddings: pd.DataFrame,
        temperature: float = 0.5,
        n_estimators: int = 100,
        max_features: str = "sqrt",
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state=None,
        verbose: int = 0,
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
        self.embeddings = embeddings
        self.temperature = temperature

    # ── Resolução de max_features → inteiro m ─────────────────────────────────

    def _resolve_m(self, n_features: int) -> int:
        mf = self.max_features
        if mf == "sqrt":      return max(1, int(np.sqrt(n_features)))
        if mf == "log2":      return max(1, int(np.log2(n_features)))
        if isinstance(mf, float): return max(1, int(mf * n_features))
        if isinstance(mf, int):   return mf
        if mf is None:            return n_features
        raise ValueError(f"max_features inválido: {mf!r}")

    # ── Constrói matriz de embeddings alinhada às features do dataset ─────────

    def _build_emb_matrix(self, feature_names: List[str]) -> np.ndarray:
        """
        Retorna array (n_features, D). Genes sem embedding têm linha NaN.
        """
        D = self.embeddings.shape[1]
        emb_idx = {g: i for i, g in enumerate(self.embeddings.index)}
        emb_vals = self.embeddings.values

        matrix = np.full((len(feature_names), D), np.nan)
        for col_i, gene in enumerate(feature_names):
            if gene in emb_idx:
                matrix[col_i] = emb_vals[emb_idx[gene]]
        return matrix

    # ── Treina uma única árvore (chamada em paralelo) ─────────────────────────

    def _fit_single_tree(self, X, y, seed):
        rng = check_random_state(seed)
        if self.bootstrap:
            idx = rng.choice(len(y), size=len(y), replace=True)
            X, y = X[idx], y[idx]

        return _EmbeddingGuidedTree(
            emb_matrix=self.emb_matrix_,
            m=self.m_,
            n_classes=self.n_classes_,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=rng.randint(0, 2**31),
            temperature=self.temperature,
        ).fit(X, y)

    # ── Interface sklearn: fit ────────────────────────────────────────────────

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            feature_names_ = list(X.columns)
            X = X.values
        else:
            raise ValueError("Passe X como pd.DataFrame para mapear nomes de genes aos embeddings.")

        self.feature_names_in_ = np.array(feature_names_)
        self.emb_matrix_       = self._build_emb_matrix(feature_names_)

        n_emb = int((~np.isnan(self.emb_matrix_[:, 0])).sum())
        if self.verbose:
            print(f"[EmbeddingGuidedRF] {n_emb}/{len(feature_names_)} genes com embedding.")

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
        return self

    # ── Interface sklearn: predict_proba / predict ────────────────────────────

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# ClusterAwareRandomForestClassifier  (mantido para compatibilidade)
# ─────────────────────────────────────────────────────────────────────────────

def _clusters_from_dataframe(
    df: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, List[int]]:
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


def _sample_features_priority(clusters, priority_key, m, rng):
    inside  = list(clusters[priority_key])
    outside = [f for k, feats in clusters.items() if k != priority_key for f in feats]
    quota_in  = (m + 1) // 2
    quota_out = m // 2
    take_in   = min(quota_in,  len(inside))
    take_out  = min(quota_out, len(outside))
    take_in   = min(take_in  + (quota_out - take_out), len(inside))
    take_out  = min(take_out + (quota_in  - take_in),  len(outside))
    selected = []
    if take_in:
        selected.extend(rng.choice(inside,  size=take_in,  replace=False).tolist())
    if take_out:
        selected.extend(rng.choice(outside, size=take_out, replace=False).tolist())
    return np.array(selected, dtype=int)


class _ClusterAwareTree:
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
        if not len(y): return 0.0
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
            if len(uniq) < 2: continue
            for thr in (uniq[:-1] + uniq[1:]) / 2.0:
                lm = vals <= thr
                rm = ~lm
                if lm.sum() < self.min_samples_leaf or rm.sum() < self.min_samples_leaf: continue
                g = self._weighted_gini(y[lm], y[rm])
                if g < best_g: best_g, best_f, best_t = g, f, thr
        return best_f, best_t

    def _build(self, X, y, depth, priority_key, rng):
        if (len(np.unique(y)) == 1 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return self._leaf(y)
        if priority_key is None:
            feat_ids = np.array(rng.choice(list(range(X.shape[1])), size=self.m, replace=False))
        else:
            feat_ids = _sample_features_priority(self.clusters, priority_key, self.m, rng)
        f, thr = self._best_split(X, y, feat_ids)
        if f is None: return self._leaf(y)
        if priority_key is None:
            priority_key = self._feature_to_cluster[f]
            self.priority_cluster_ = priority_key
        lm = X[:, f] <= thr
        return {"f": f, "thr": thr,
                "left":  self._build(X[lm],  y[lm],  depth + 1, priority_key, rng),
                "right": self._build(X[~lm], y[~lm], depth + 1, priority_key, rng)}

    def _leaf(self, y):
        p = np.zeros(self.n_classes)
        if len(y): p = np.bincount(y, minlength=self.n_classes) / len(y)
        return {"proba": p}

    def fit(self, X, y):
        self._feature_to_cluster = {feat: key for key, feats in self.clusters.items() for feat in feats}
        self.priority_cluster_ = None
        self._root = self._build(X, y, depth=0, priority_key=None,
                                 rng=check_random_state(self.random_state))
        return self

    def _walk(self, x, node):
        if "proba" in node: return node["proba"]
        return self._walk(x, node["left"] if x[node["f"]] <= node["thr"] else node["right"])

    def predict_proba(self, X):
        return np.array([self._walk(x, self._root) for x in X])


class ClusterAwareRandomForestClassifier(RandomForestClassifier):
    """Random Forest guiado por clusters de ontologias (v1 — mantido para compatibilidade)."""

    def __init__(self, clusters: pd.DataFrame, feature_names=None,
                 n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 n_jobs=1, random_state=None, verbose=0):
        super().__init__(n_estimators=n_estimators, max_features=max_features,
                         max_depth=max_depth, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, bootstrap=bootstrap,
                         n_jobs=n_jobs, random_state=random_state, verbose=verbose)
        self.clusters = clusters
        self.feature_names = feature_names

    def _resolve_m(self, n_features):
        mf = self.max_features
        if mf == "sqrt":      return max(1, int(np.sqrt(n_features)))
        if mf == "log2":      return max(1, int(np.log2(n_features)))
        if isinstance(mf, float): return max(1, int(mf * n_features))
        if isinstance(mf, int):   return mf
        if mf is None:            return n_features
        raise ValueError(f"max_features inválido: {mf!r}")

    def _fit_single_tree(self, X, y, seed):
        rng = check_random_state(seed)
        if self.bootstrap:
            idx = rng.choice(len(y), size=len(y), replace=True)
            X, y = X[idx], y[idx]
        return _ClusterAwareTree(clusters=self.clusters_, m=self.m_, n_classes=self.n_classes_,
                                  max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                  min_samples_leaf=self.min_samples_leaf,
                                  random_state=rng.randint(0, 2**31)).fit(X, y)

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
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1
        self.n_features_in_ = X.shape[1]
        self.m_ = self._resolve_m(X.shape[1])
        y_enc = np.searchsorted(self.classes_, y)
        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, 2**31, size=self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_single_tree)(X, y_enc, int(s)) for s in seeds)
        self.tree_priority_clusters_ = [t.priority_cluster_ for t in self.estimators_]
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame): X = X.values
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
