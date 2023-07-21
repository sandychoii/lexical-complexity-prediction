import pandas as pd

tagged_dir = 'bnc_cefr_tagged'
save_dir = 'bnc_filtered'


# (POS, c5) to keep as UNK
keep_pairs = [('UNC', 'UNC'), ('SUBST', 'NP0'), ('INTERJ', 'ITJ'), ('SUBST', 'ZZ0')]
keep_pairs += [('STOP', c5) for c5 in ['POS', 'PUL', 'PUN', 'PUQ', 'PUR']]

fnames = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]

def filter_unk(tagged_dir, save_dir, fnames):
    for fname in fnames:
        df = pd.read_csv(f'{tagged_dir}/{fname}_tagged.tsv', sep='\t')

        print(f'Processing {fname}_tagged.tsv...')
    
        # 'UNK' but POS and c5 pairs are not in the keep_pairs
        mask = (df['CEFR'] == 'UNK') & (~df[['POS', 'c5']].apply(tuple, axis=1).isin(keep_pairs))

        sentences_to_remove = df.loc[mask, 'SentenceID'].unique()
        df_filtered = df[~df['SentenceID'].isin(sentences_to_remove)]

        df_filtered.to_csv(f'{save_dir}/{fname}_filtered.tsv', sep='\t', index=False)
        print(f"Saved {save_dir}/{fname}_filtered.tsv.")