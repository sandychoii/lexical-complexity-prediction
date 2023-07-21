import pandas as pd
import json
import pdb


# Conversion dictionary
conversion_dict = {
    "adverb": "ADV",
    "number": "ADJ",  # all CRD (Cardinal number is ADJ)
    "adjective": "ADJ",
    "adv": "ADV",
    "determiner": {"DPS": "PRON", "DTQ": "PRON", # DPS, DTQ (PRON) # DT0 (ADJ)
                   "DT0": "ADJ", "AT0": "ART", "AJ0":"ADJ"},
    "preposition": "PREP",
    "noun": "SUBST",
    "modal verb": "VERB",  # VM0
    "verb": "VERB",
    "exclamation": "INTERJ",
    "ordinal number": "ADJ",
    "auxiliary verb": "VERB",  # be, do, have
    "conjunction": {"CJT":"CONJ", "CJC":"CONJ", "CJS": "CONJ", "DT0-CJT": "ADJ"}, # latter is for conjunction 'that'
    "infinitive marker": "PREP",  # to
    "pronoun": "PRON",
    "adverb; preposition": ["PREP", "ADV", "CONJ"],  # as -> PREP or CONJ or ADV
}

extended_conversion_dict = {
    'interjection': 'INTERJ',
    'preposition': 'PREP',
    'noun': 'SUBST',
    'adverb': 'ADV',
    'modal auxiliary': 'VERB',
    'do-verb': 'VERB',
    'have-verb': 'VERB',
    'verb': 'VERB',
    'number': 'ADJ',
    'adjective': 'ADJ',
    'be-verb': 'VERB',
    'infinitive-to': 'PREP',
    'conjunction': {"CJT":"CONJ", "CJC":"CONJ", "CJS": "CONJ", "DT0-CJT": "ADJ"},
    'determiner': {"DPS": "PRON", "DTQ": "PRON", # DPS, DTQ (PRON) # DT0 (ADJ)
                   "DT0": "ADJ", "AT0": "ART", "AJ0":"ADJ"},
    'vern': 'vern',
    'pronoun': 'pronoun'
}


exception_in_bnc = {'than': ('than', 'CONJ', 'CJS'),}


# Create a dictionary where keys are tuples (lemma, pos) and values are CEFR levels.
def create_cefr_dict(cefr_wordlist_path):
    print("Creating CEFR dictionary...")
    # load cefr json file
    with open(cefr_wordlist_path, 'r') as file:
        cefr_wordlist = json.load(file)

    cefr_dict = {}
    for word, entries in cefr_wordlist.items():
        word = word.lower()
        for entry in entries:
            cefr_pos = entry['pos']

            # handle exceptions that bnc defines differently 
            if word in exception_in_bnc:
                key = exception_in_bnc[word]
                if key not in cefr_dict:
                    cefr_dict[key] = []
                cefr_dict[key].append(entry['level'])

            elif cefr_pos in conversion_dict:

                if isinstance(conversion_dict[cefr_pos], str):  # For normal case
                    key = (word, conversion_dict[cefr_pos])
                    if key not in cefr_dict:
                        cefr_dict[key] = []
                    cefr_dict[key].append(entry['level'])

                elif isinstance(conversion_dict[cefr_pos], list):  # For multiple cases of (lemma, pos) # as
                    for pos in conversion_dict[cefr_pos]:
                        key = (word, pos)
                        if key not in cefr_dict:
                            cefr_dict[key] = []
                        cefr_dict[key].append(entry['level'])

                else:  # For special cases, (lemma, pos, c5) for key # determiner
                    for conv_key, conv_val in conversion_dict[cefr_pos].items():
                        key = (word, conv_val, conv_key)
                        if key not in cefr_dict:
                            cefr_dict[key] = []
                        cefr_dict[key].append(entry['level'])

    # Exceptions for words not in the CEFR wordlist.
    cefr_dict[('there', 'PRON')] = ['A1']
    cefr_dict[("'s", 'UNC')] = ['A1']
    cefr_dict[('ought', 'VERB')] = ['A1']
    cefr_dict[('shall', 'VM0')] = ['A1']

    return cefr_dict


def create_cefr_dict_from_extended(cefr_dict, extended_wordlist_path):
    print("Creating CEFR dictionary from extended wordlist...")

    ## for CEFR-J, octanove
    # load csv
    df = pd.read_csv(extended_wordlist_path)

    df['word_pos'] = list(zip(df['headword'], df['pos']))
    df = df[~df['word_pos'].isin(cefr_dict)]

    extended_dict = {}
    for _, row in df.iterrows():
        word = row['headword']
        pos = row['pos']
        cefr_level = row['CEFR']

        if isinstance(extended_conversion_dict[pos], str):
            key = (word, extended_conversion_dict[pos])
            if key not in extended_dict:
                extended_dict[key] = []
            extended_dict[key].append(cefr_level)
        elif isinstance(extended_conversion_dict, dict):
            for conv_key, conv_val in extended_conversion_dict[pos].items():
                key = (word, conv_val, conv_key)
                if key not in extended_dict:
                    extended_dict[key] = []
                extended_dict[key].append(cefr_level)
    
    extended_dict = {k: v for k, v in extended_dict.items() if k not in cefr_dict}

    return extended_dict



def compare_cefr_levels(levels):
    if "UNK" in levels:
        return "UNK"
    # Order of CEFR levels from lowest to highest.
    order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    # Find and return the lowest level. => Insert WSD module here
    return min(levels, key=order.index)


def handle_rows(row, cefr_dict, extended_cefr_dict=None):
    lemma = row['Lemma']
    pos = row['POS']
    c5 = row['c5']

    if (lemma, pos) in cefr_dict:
        level_list = cefr_dict[(lemma, pos)]
    elif (lemma, pos, c5) in cefr_dict:
        level_list = cefr_dict[(lemma, pos, c5)]
    elif extended_cefr_dict and (lemma, pos) in extended_cefr_dict:
        level_list = extended_cefr_dict[(lemma, pos)]
    elif extended_cefr_dict and (lemma, pos, c5) in extended_cefr_dict:
        level_list = extended_cefr_dict[(lemma, pos, c5)]
    else:
        level_list = ['UNK']

    return compare_cefr_levels(level_list)


def tag_cefr_level(cefr_wordlist_path, extended_wordlist_path, fnames, processed_dir, cefr_tagged_dir):
    cefr_dict = create_cefr_dict(cefr_wordlist_path=cefr_wordlist_path)
    if extended_wordlist_path:
        extended_cefr_dict = create_cefr_dict_from_extended(cefr_dict.keys(), extended_wordlist_path)
        
    for fname in fnames:
        # load tsv file
        print(f'Processing {fname}.tsv...')
        df = pd.read_csv(f'{processed_dir}/{fname}.tsv', sep='\t')
        print(df.head())
        if extended_wordlist_path:
            # Use the dictionary to add a new column 'CEFR' to the DataFrame.
            df['CEFR'] = df.apply(handle_rows, args=(cefr_dict, extended_cefr_dict), axis=1)

        else:
            # Use the dictionary to add a new column 'CEFR' to the DataFrame.
            df['CEFR'] = df.apply(
                lambda row: compare_cefr_levels(
                    cefr_dict.get((row['Lemma'], row['POS']), cefr_dict.get((row['Lemma'], row['POS'], row['c5']), ["UNK"]))
                ), axis=1
            )

        df.to_csv(f'{cefr_tagged_dir}/{fname}_tagged.tsv', sep='\t', index=False)
        print(f'Saved {cefr_tagged_dir}/{fname}_tagged.tsv.')
    

if __name__ == '__main__':
    fnames = ["A"]
    cefr_dict = create_cefr_dict(cefr_wordlist_path='cefr/amer_word_single_entry.json')

    for fname in fnames:
        # load tsv file
        print(f'Processing {fname}.tsv...')
        df = pd.read_csv(f'bnc_processed/{fname}.tsv', sep='\t')

        # Use the dictionary to add a new column 'CEFR' to the DataFrame.
        df['CEFR'] = df.apply(
            lambda row: compare_cefr_levels(
                cefr_dict.get((row['Lemma'], row['POS']), cefr_dict.get((row['Lemma'], row['POS'], row['c5']), ["UNK"]))
            ), axis=1
        )

        df.to_csv(f'bnc_cefr_tagged/{fname}_tagged.tsv', sep='\t', index=False)
        print(f'Saved bnc_cefr_tagged/{fname}_tagged.tsv.')
