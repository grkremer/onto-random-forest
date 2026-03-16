"""
gene_clustering.py
==================
Geração de embeddings de genes via mOWL + OWL2Vec* + Gene Ontology (GO)
e clusterização com KMeans e HDBSCAN.

Instalação das dependências:
    pip install mowl-borg mygene gensim scikit-learn umap-learn hdbscan matplotlib

Requer Java JDK instalado (OWLAPI roda na JVM).

Uso:
    python gene_clustering.py BLCA BRCA LIHC PRAD
    # ou, dentro de outro script:
    from gene_clustering import gera_clusters_genes_com_mowl
    gera_clusters_genes_com_mowl("BRCA")
"""

from __future__ import annotations

import mowl
mowl.init_jvm("8g")  # deve ser a primeira chamada — só pode ser feita uma vez

import argparse
import logging
import os
import requests
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import mygene
import numpy as np
import pandas as pd
import umap
from gensim.models import Word2Vec
from mowl.corpus import (
    extract_and_save_annotation_corpus,
    extract_and_save_axiom_corpus,
)
from mowl.datasets import PathDataset
from mowl.projection import OWL2VecStarProjector
from mowl.walking import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes de configuração (podem ser sobrescritas via argumentos)
# ---------------------------------------------------------------------------
# Entrada
INPUT_DIR      = Path("./data/input")
# Ontologia GO (arquivo grande, compartilhado entre todos os datasets)
ONTOLOGY_DIR   = Path("./data/ontology")
# Cache de arquivos intermediários de treino (regeneráveis)
CACHE_DIR      = Path("./data/cache")
# Saídas
OUT_EMBEDDINGS = Path("./data/output/embeddings")
OUT_CLUSTERS   = Path("./data/output/clusters")
OUT_UMAP       = Path("./data/output/umap")

GO_OWL_URL    = "http://purl.obolibrary.org/obo/go.owl"

W2V_VECTOR_SIZE = 128
W2V_WINDOW      = 5
W2V_MIN_COUNT   = 1
W2V_EPOCHS      = 50

NODE2VEC_NUM_WALKS   = 20
NODE2VEC_WALK_LENGTH = 10
NODE2VEC_P           = 1.0
NODE2VEC_Q           = 1.0

KMEANS_K_RANGE   = range(2, 20)
HDBSCAN_MIN_SAMPLES = 3


# ===========================================================================
# 1. Download da ontologia GO
# ===========================================================================

def download_go_owl(dest: Path) -> None:
    """Baixa o arquivo go.owl se ainda não existir."""
    if dest.exists():
        log.info("go.owl já existe em %s", dest)
        return
    log.info("Baixando go.owl (~200 MB) de %s …", GO_OWL_URL)
    with requests.get(GO_OWL_URL, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=16_384):
                f.write(chunk)
    log.info("Download concluído: %s", dest)


# ===========================================================================
# 2. Busca de termos GO por gene (mygene.info)
# ===========================================================================

def fetch_go_terms(gene_list: list[str]) -> dict[str, list[str]]:
    """
    Consulta mygene.info e retorna um mapa {gene_symbol: [go_uri, …]}.
    Apenas as namespaces BP, MF e CC são consideradas.
    """
    mg = mygene.MyGeneInfo()
    hits = mg.querymany(
        gene_list,
        scopes="symbol",
        fields="go",
        species="human",
        returnall=False,
        verbose=False,
    )
    gene_go_map: dict[str, list[str]] = {}
    for hit in hits:
        symbol = hit.get("query", "")
        entries = hit.get("go", {})
        uris: list[str] = []
        for ns in ("BP", "MF", "CC"):
            items = entries.get(ns, [])
            if isinstance(items, dict):   # mygene retorna dict quando há só 1 termo
                items = [items]
            for item in items:
                go_id = item.get("id", "")
                if go_id:
                    # URI no formato que mOWL usa: http://purl.obolibrary.org/obo/GO_XXXXXXX
                    uri = f"http://purl.obolibrary.org/obo/{go_id.replace(':', '_')}"
                    uris.append(uri)
        gene_go_map[symbol] = list(set(uris))
    return gene_go_map


# ===========================================================================
# 3. Projeção OWL → Grafo + Random Walks + Corpus de Anotações
# ===========================================================================

class CombinedSentences:
    """Itera sobre walks (estrutural) e corpus de anotações (textual)."""

    def __init__(self, walks_file: str, corpus_file: str) -> None:
        self.files = [walks_file, corpus_file]

    def __iter__(self):
        for fpath in self.files:
            with open(fpath, encoding="utf-8") as fh:
                for line in fh:
                    yield line.strip().split()


def build_owl2vec_model(
    go_owl_file: str,
    walks_file: str,
    corpus_file: str,
) -> Word2Vec:
    """
    Constrói um modelo Word2Vec OWL2Vec* a partir da ontologia GO.

    Passos:
        1. Projeta a ontologia OWL em um grafo (OWL2VecStarProjector).
        2. Gera random walks no grafo (Node2Vec).
        3. Extrai corpus textual de anotações (axiomas + rótulos).
        4. Treina Word2Vec combinando walks e corpus.
    """
    dataset = PathDataset(go_owl_file)

    # --- Projeção OWL → grafo ---
    projector = OWL2VecStarProjector(
        bidirectional_taxonomy=True,   # gera arestas SubClass E SuperClass
        only_taxonomy=False,           # inclui relações além de is_a (part_of, etc.)
        include_literals=True,         # inclui anotações textuais no grafo
    )
    log.info("Projetando ontologia GO em grafo …")
    edges = projector.project(dataset.ontology)

    # --- Random Walks (Node2Vec) ---
    walker = Node2Vec(
        num_walks=NODE2VEC_NUM_WALKS,
        walk_length=NODE2VEC_WALK_LENGTH,
        p=NODE2VEC_P,
        q=NODE2VEC_Q,
        outfile=walks_file,
        workers=os.cpu_count(),
    )
    log.info("Gerando random walks …")
    walker.walk(edges)

    # --- Corpus de anotações ---
    log.info("Extraindo corpus de anotações …")
    extract_and_save_axiom_corpus(dataset.ontology, corpus_file)
    extract_and_save_annotation_corpus(dataset.ontology, corpus_file, mode="a")

    # --- Treinamento Word2Vec ---
    log.info("Treinando Word2Vec (OWL2Vec*) …")
    sentences = CombinedSentences(walks_file, corpus_file)
    model = Word2Vec(
        sentences,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=os.cpu_count(),
        epochs=W2V_EPOCHS,
        sg=1,   # skip-gram (recomendado para ontologias esparsas)
    )
    log.info("Vocabulário treinado: %d entidades/tokens", len(model.wv))
    return model


# ===========================================================================
# 4. Geração de embeddings por gene
# ===========================================================================

def compute_gene_embeddings(
    annotated: dict[str, list[str]],
    wv,
) -> tuple[list[str], np.ndarray]:
    """
    Calcula o embedding de cada gene como a média dos embeddings dos seus
    termos GO presentes no vocabulário do modelo Word2Vec.

    Retorna:
        gene_names : lista de nomes dos genes com embedding válido
        X          : array (n_genes, vector_size) com os embeddings
    """
    gene_names: list[str] = []
    gene_vectors: list[np.ndarray] = []

    for gene, uris in annotated.items():
        vecs = [wv[uri] for uri in uris if uri in wv]
        if vecs:
            gene_names.append(gene)
            gene_vectors.append(np.mean(vecs, axis=0))

    log.info("Genes com embedding válido: %d", len(gene_names))
    return gene_names, np.array(gene_vectors)


# ===========================================================================
# 5. Clusterização
# ===========================================================================

def select_best_kmeans(
    X: np.ndarray,
    k_range: range = KMEANS_K_RANGE,
) -> tuple[KMeans, int, list[float]]:
    """
    Testa diferentes valores de K e seleciona o melhor KMeans pelo
    coeficiente de silhueta.

    Retorna:
        best_model : KMeans treinado com o melhor K
        best_k     : valor de K selecionado
        scores     : coeficientes de silhueta para cada K testado
    """
    best_k, best_score, best_model = 2, -1.0, None
    scores: list[float] = []
    for k in k_range:
        km     = KMeans(n_clusters=k, n_init="auto")
        labels = km.fit_predict(X)
        score  = silhouette_score(X, labels)
        scores.append(score)
        if score > best_score:
            best_score, best_k, best_model = score, k, km
    log.info("Melhor K=%d | Silhouette=%.4f", best_k, best_score)
    return best_model, best_k, scores


def run_hdbscan(X: np.ndarray) -> np.ndarray:
    """
    Executa HDBSCAN e retorna os rótulos de cluster.
    Genes não classificados recebem rótulo -1.
    """
    hdb = hdbscan.HDBSCAN(min_samples=HDBSCAN_MIN_SAMPLES, metric="euclidean")
    labels = hdb.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise      = (labels == -1).sum()
    log.info("HDBSCAN: %d clusters | %d genes sem cluster (ruído)", n_clusters, noise)
    return labels


# ===========================================================================
# 6. Visualização (UMAP 2D)
# ===========================================================================

def plot_clusters(
    X_scaled: np.ndarray,
    gene_names: list[str],
    kmeans_labels: np.ndarray,
    hdb_labels: np.ndarray,
    best_k: int,
    output_path: str,
) -> np.ndarray:
    """
    Reduz para 2D com UMAP e gera um gráfico comparando KMeans e HDBSCAN.
    Salva a figura em *output_path* e retorna as coordenadas 2D.
    """
    n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d    = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, labels, title in zip(
        axes,
        [kmeans_labels, hdb_labels],
        [f"KMeans  k={best_k}", f"HDBSCAN  {n_hdb} clusters"],
    ):
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab20", s=18, alpha=0.85)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.suptitle("Clusterização de Genes — mOWL + OWL2Vec* + GO", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    log.info("Figura salva em %s", output_path)
    plt.show()
    return X_2d


# ===========================================================================
# 7. Pipeline principal
# ===========================================================================

def gera_clusters_genes_com_mowl(data_name: str) -> None:
    """
    Pipeline completo de geração de embeddings e clusterização de genes.

    Parâmetros
    ----------
    data_name : str
        Nome do conjunto de dados (ex.: "BRCA", "PRAD"). Deve corresponder
        a um diretório em data/input/deepkegg/<data_name>_data/.
    """
    log.info("=== Iniciando pipeline para %s ===", data_name)

    # --- Garantir que os diretórios existem ---
    for d in (ONTOLOGY_DIR, CACHE_DIR, OUT_EMBEDDINGS, OUT_CLUSTERS, OUT_UMAP):
        d.mkdir(parents=True, exist_ok=True)

    go_owl_file = str(ONTOLOGY_DIR / "go.owl")
    walks_file  = str(CACHE_DIR    / "walks.txt")
    corpus_file = str(CACHE_DIR    / "corpus.txt")

    # --- Leitura dos genes de interesse ---
    mRNA_path = INPUT_DIR / "deepkegg" / f"{data_name}_data" / "mRNA_data.csv"
    mRNA_data = pd.read_csv(mRNA_path)
    genes = mRNA_data.columns[1:].tolist()   # ignora a coluna 'Case_ID'
    log.info("Total de genes carregados: %d", len(genes))

    # --- Etapa 1: Download da ontologia ---
    download_go_owl(ONTOLOGY_DIR / "go.owl")

    # --- Etapa 2: Busca de anotações GO ---
    log.info("Buscando anotações GO …")
    gene_go_map = fetch_go_terms(genes)
    annotated   = {g: t for g, t in gene_go_map.items() if t}
    log.info("Genes anotados: %d / %d", len(annotated), len(genes))

    # --- Etapa 3: Embeddings OWL2Vec* ---
    w2v_model  = build_owl2vec_model(go_owl_file, walks_file, corpus_file)
    gene_names, X = compute_gene_embeddings(annotated, w2v_model.wv)

    # Salva embeddings brutos
    emb_path = str(OUT_EMBEDDINGS / f"{data_name}_embeddings_mowl.csv")
    pd.DataFrame(X, index=gene_names).to_csv(emb_path)
    log.info("Embeddings salvos em %s", emb_path)

    # --- Etapa 4: Normalização ---
    Xs = StandardScaler().fit_transform(X)

    # --- Etapa 5: Clusterização ---
    km_model, best_k, _ = select_best_kmeans(Xs)
    kmeans_labels = km_model.labels_
    hdb_labels    = run_hdbscan(Xs)

    # --- Etapa 6: Visualização UMAP ---
    plot_path = str(OUT_CLUSTERS / f"clusters_{data_name}.png")
    X_2d = plot_clusters(Xs, gene_names, kmeans_labels, hdb_labels, best_k, plot_path)

    # Salva coordenadas UMAP
    umap_path = str(OUT_UMAP / f"{data_name}_umap_mowl.csv")
    pd.DataFrame(X_2d, index=gene_names, columns=["UMAP1", "UMAP2"]).to_csv(umap_path)
    log.info("Coordenadas UMAP salvas em %s", umap_path)

    # --- Etapa 7: Exportar resultados de cluster ---
    results_path = str(OUT_CLUSTERS / f"genes_clustered_{data_name}.csv")
    df = pd.DataFrame({
        "gene":            gene_names,
        "cluster_kmeans":  kmeans_labels,
        "cluster_hdbscan": hdb_labels,
    })
    df.to_csv(results_path, index=False)
    log.info("Resultados exportados em %s", results_path)

    # Resumo dos maiores clusters KMeans
    print("\nTop clusters KMeans:")
    print(df.groupby("cluster_kmeans")["gene"].count().sort_values(ascending=False).head(10))
    log.info("=== Pipeline concluído para %s ===\n", data_name)


# ===========================================================================
# 8. Ponto de entrada via linha de comando
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera embeddings de genes com mOWL + OWL2Vec* e clusteriza.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        metavar="DATASET",
        help="Nome(s) do(s) conjunto(s) de dados (ex.: BLCA BRCA LIHC PRAD).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    for ds in args.datasets:
        print(f"\nGenerating clusters for {ds}")
        gera_clusters_genes_com_mowl(ds)
