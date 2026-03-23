import pandas as pd

datasets = ['BLCA', 'BRCA', 'LIHC', 'PRAD']

summary = []
missing_counts = {}
mismatch_examples = {}

for ds in datasets:
    dkk  = pd.read_csv(f'data/input/deepkegg/{ds}_data/response.csv')
    full = pd.read_csv(f'data/input/tcga_full/{ds}_data/response.csv')

    # Dedup no full (barcodes de 12 chars podem ter colisão)
    full_dedup = full.drop_duplicates(subset='Case_ID', keep='first')

    merged = dkk.merge(full_dedup, on='Case_ID', suffixes=('_deepkegg', '_full'))
    match  = (merged['response_deepkegg'] == merged['response_full']).sum()
    total  = len(merged)
    div    = total - match
    pct    = 100 * match / total if total else 0.0

    missing = [x for x in dkk['Case_ID'] if x not in set(full['Case_ID'])]

    summary.append({
        'Dataset': ds,
        'IDs_deepkegg': len(dkk),
        'IDs_comuns': total,
        'Labels_iguais': match,
        'Divergencias': div,
        'Acordo_%': round(pct, 1),
        'Missing_no_full': len(missing),
    })

    mismatch_examples[ds] = merged[merged['response_deepkegg'] != merged['response_full']][
        ['Case_ID','response_deepkegg','response_full']
    ].head(5)

df_summary = pd.DataFrame(summary)
print("=" * 70)
print("RESUMO DE CONSISTÊNCIA: DeepKEGG vs TCGA Full (PFI)")
print("=" * 70)
print(df_summary.to_string(index=False))

print()
for ds in datasets:
    mm = mismatch_examples[ds]
    if len(mm):
        print(f"[{ds}] Exemplos de divergência:")
        print(mm.to_string(index=False))
        print()
