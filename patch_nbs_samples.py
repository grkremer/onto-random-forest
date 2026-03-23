import json

def patch_notebook(filepath):
    try:
        with open(filepath, encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        return False
        
    patched = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = []
            needs_patch = False
            for line in cell['source']:
                if "X_raw         = mRNA_data.drop(columns=['Case_ID'])" in line:
                    needs_patch = True
            
            if needs_patch:
                patched = True
                new_source = []
                for line in cell['source']:
                    if "X_raw         = mRNA_data.drop(columns=['Case_ID'])" in line:
                        new_source.append("    # Mesclar X e y pelo Case_ID para evitar divergências de amostras\n")
                        new_source.append("    df_merged     = pd.merge(mRNA_data, response_data, on='Case_ID')\n")
                        new_source.append("    X_raw         = df_merged.drop(columns=['Case_ID', 'response'])\n")
                    elif "y             = response_data['response'].values" in line:
                        new_source.append("    y             = df_merged['response'].values\n")
                    elif "y = response_data['response'].values" in line:
                        new_source.append("    y = df_merged['response'].values\n")
                    elif "X = mRNA_data.drop(columns=['Case_ID'])" in line:
                        new_source.append("    df_merged = pd.merge(mRNA_data, response_data, on='Case_ID')\n")
                        new_source.append("    X = df_merged.drop(columns=['Case_ID', 'response'])\n")
                    elif "y = response_data['response']" in line and not "values" in line:
                        new_source.append("    y = df_merged['response']\n")
                    else:
                        # try plot_importance specific variables:
                        if "X = mRNA_data.drop(columns=['Case_ID'])" in line:
                            new_source.append("    df_merged = pd.merge(mRNA_data, response_data, on='Case_ID')\n")
                            new_source.append("    X = df_merged.drop(columns=['Case_ID', 'response'])\n")
                        else:
                            new_source.append(line)
                            
                # Check for other variations in plot_importance
                if "X = mRNA_data.drop(columns=['Case_ID'])" not in "".join(new_source):
                    new_source2 = []
                    for line in new_source:
                        if "X = mRNA_data.drop(columns=['Case_ID'])" in line:
                            new_source2.append("    df_merged = pd.merge(mRNA_data, response_data, on='Case_ID')\n")
                            new_source2.append("    X = df_merged.drop(columns=['Case_ID', 'response'])\n")
                        elif "y = response_data['response'].values" in line:
                            new_source2.append("    y = df_merged['response'].values\n")
                        else:
                            new_source2.append(line)
                    new_source = new_source2
                            
                cell['source'] = new_source
                
    if patched:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Patched {filepath} successfully.")
    else:
        print(f"No changes needed for {filepath} or patterns didn't match.")
    return patched

# Patch test_ontorf
patch_notebook('test_ontorf.ipynb')

# For plot_importance_umap we need a slightly broader replacement because variables might be named differently
# Let's read it specifically
with open('plot_importance_umap.ipynb', encoding='utf-8') as f:
    nb2 = json.load(f)
    
changed2 = False
for cell in nb2.get('cells', []):
    if cell.get('cell_type') == 'code':
        txt = "".join(cell['source'])
        if "X = mRNA_data.drop(columns=['Case_ID'])" in txt or "y = response_data['response']" in txt:
            new_src = []
            for line in cell['source']:
                if "X = mRNA_data.drop(columns=['Case_ID'])" in line:
                    new_src.append("    df_merged = pd.merge(mRNA_data, response_data.drop_duplicates(subset='Case_ID', keep='first'), on='Case_ID')\n")
                    new_src.append("    X = df_merged.drop(columns=['Case_ID', 'response'])\n")
                    changed2 = True
                elif "y = response_data['response'].values" in line:
                    new_src.append("    y = df_merged['response'].values\n")
                elif "y = response_data['response']" in line:
                    new_src.append("    y = df_merged['response'].values\n")
                else:
                    new_src.append(line)
            cell['source'] = new_src
            
if changed2:
    with open('plot_importance_umap.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb2, f, indent=1)
    print("Patched plot_importance_umap.ipynb successfully.")
