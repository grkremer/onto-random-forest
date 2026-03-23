"""
download_tcga.py
----------------
Baixa os datasets completos do TCGA a partir do UCSC Xena (TCGA Hub),
transforma para o formato usado pelo DeepKEGG e salva em
data/input/tcga_full/{DATASET}_data/
"""

import os
import io
import gzip
import requests
import mygene
import pandas as pd
import numpy as np

# ── Configuração ──────────────────────────────────────────────────────────────
DATASETS = ["BLCA", "BRCA", "LIHC", "PRAD"]

# Legacy TCGA HiSeqV2 (RSEM, gene symbol index) – todos retornam 200
MRNA_URL = "https://tcga.xenahubs.net/download/TCGA.{ds}.sampleMap/HiSeqV2.gz"

# Pan-Cancer survival: PFI labels
SURVIVAL_URL = "https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp"

OUTPUT_ROOT = "data/input/tcga_full"

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_gz_tsv(url: str, label: str) -> pd.DataFrame:
    """Download a .gz TSV URL and return as a DataFrame."""
    print(f"  Baixando {label} …")
    r = requests.get(url, timeout=300, stream=True, allow_redirects=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    downloaded = 0
    for chunk in r.iter_content(chunk_size=1024 * 512):
        buf.write(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"    {pct:.0f}% ({downloaded // 1024 // 1024} MB / {total // 1024 // 1024} MB)", end="\r")
    print()
    buf.seek(0)
    with gzip.open(buf, "rt") as f:
        return pd.read_csv(f, sep="\t", index_col=0)


def download_tsv(url: str, label: str) -> pd.DataFrame:
    """Download a plain TSV URL and return as a DataFrame."""
    print(f"  Baixando {label} …")
    r = requests.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep="\t", index_col=0)


# ── Download do arquivo de Sobrevivência uma única vez ────────────────────────

print("=" * 60)
print("Baixando arquivo de sobrevivência Pan-Cancer (PFI) …")
print("=" * 60)

survival_raw = download_tsv(SURVIVAL_URL, "Survival Pan-Cancer")
print("  Colunas disponíveis:", list(survival_raw.columns)[:10], "…")

# ── Loop por Dataset ──────────────────────────────────────────────────────────

for DATASET in DATASETS:
    print(f"\n{'='*60}")
    print(f"  Processando {DATASET}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_ROOT, f"{DATASET}_data")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Baixa mRNA HiSeqV2 (gene symbols no índice, amostras nas colunas) ─
    url = MRNA_URL.format(ds=DATASET)
    df_mrna = download_gz_tsv(url, f"{DATASET} mRNA HiSeqV2")
    print(f"  Shape bruta (genes x amostras): {df_mrna.shape}")
    # O índice do HiSeqV2 já é gene symbol – não precisa de mygene!
    print(f"  Exemplo de gene IDs: {list(df_mrna.index[:5])}")

    # ── 2. Remove genes duplicados: mantém o de maior variância ──────────────
    df_mrna = df_mrna[~df_mrna.index.duplicated(keep='first')]

    # ── 3. Transpõe → amostras x genes (igual ao DeepKEGG) ──────────────────
    df_mrna_T = df_mrna.T  # amostras x genes
    # Normaliza barcodes para 12 chars (TCGA-XX-XXXX)
    df_mrna_T.index = [s[:12] for s in df_mrna_T.index]
    df_mrna_T.index.name = "Case_ID"
    df_mrna_T = df_mrna_T.reset_index()
    print(f"  Shape após transposição (amostras x genes): {df_mrna_T.shape}")

    # ── 4. Gera response.csv a partir do PFI ─────────────────────────────────
    surv_ds = survival_raw[survival_raw["cancer type abbreviation"] == DATASET].copy()
    surv_ds.index = [s[:12] for s in surv_ds.index]
    surv_ds = surv_ds[["PFI"]].copy()
    surv_ds.columns = ["response"]
    surv_ds = surv_ds.dropna(subset=["response"])
    surv_ds["response"] = surv_ds["response"].astype(int)
    surv_ds.index.name = "Case_ID"
    surv_ds = surv_ds.reset_index()
    print(f"  Amostras com PFI: {len(surv_ds)} | dist: {surv_ds['response'].value_counts().to_dict()}")

    # ── 5. Intersecção ────────────────────────────────────────────────────────
    common = set(df_mrna_T["Case_ID"]) & set(surv_ds["Case_ID"])
    df_mrna_final = df_mrna_T[df_mrna_T["Case_ID"].isin(common)].reset_index(drop=True)
    df_response   = surv_ds[surv_ds["Case_ID"].isin(common)].reset_index(drop=True)
    print(f"  Amostras comuns (mRNA ∩ survival): {len(common)}")

    # ── 6. Salva ──────────────────────────────────────────────────────────────
    mrna_path    = os.path.join(out_dir, "mRNA_data.csv")
    response_path = os.path.join(out_dir, "response.csv")

    df_mrna_final.to_csv(mrna_path, index=False)
    df_response.to_csv(response_path, index=False)

    sz = os.path.getsize(mrna_path) // 1024 // 1024
    print(f"  ✓ Salvo: {mrna_path}  ({sz} MB)")
    print(f"  ✓ Salvo: {response_path}")

print(f"\n\n{'='*60}")
print("TCGA HiSeqV2 processado e salvo com sucesso em data/input/tcga_full/")
print(f"{'='*60}")
