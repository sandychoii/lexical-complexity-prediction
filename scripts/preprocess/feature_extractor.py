'''
Extracting features for given word list.
Need to specify the features for experiment.

Expected input: cefr, noad wordlist

@jychoi
'''
import syllables
from nltk.corpus import wordnet
import os, sys
import json
import pandas as pd

FEATURE_DIR = '../data/features'
FEATURES_TO_USE = {}

class FeatureExtractor:
    def __init__(self, wordlist, feature_dir, features_to_use):
        
        self.wordlist = [w.lower() for w in wordlist]
        self.features_to_use = features_to_use
        self.feature_dir = feature_dir
        self.feature_methods = {
            'surface_features': self.surface_features,
            'wordnet_features': self.wordnet_features,
            'extract_from_file': self.extract_from_file,
        }


    def surface_features(self, _):
        word_length = [len(w) for w in self.wordlist]
        syllable_length = [syllables.estimate(w) for w in self.wordlist]
        return {'word_length': word_length, 'syllable_length': syllable_length}
    

    def wordnet_features(self, _):
        num_synsets = []
        num_hypernyms = []
        num_hyponyms = []

        for word in self.wordlist:
            synsets = wordnet.synsets(word)
            num_synsets += [len(synsets)]

            hyper_num = 0
            hypon_num = 0
            for syn in synsets:
                if syn.lemmas()[0].name() == word: # synset of current word
                    hyper_num += len(syn.hypernyms())
                    hypon_num += len(syn.hyponyms())
            num_hypernyms += [hyper_num]
            num_hyponyms += [hypon_num]
        return {"num_synsets": num_synsets, "num_hypernyms": num_hypernyms, "num_hyponyms": num_hyponyms}


    def extract_from_file(self, fname2features):
        all_features = {}
        for fname, feature_list in fname2features.items():
            for ft in feature_list:
                print("Extracting", ft, "from", fname)
            extension = os.path.splitext(fname)[-1]
            if extension == '.csv':
                delimiter = ','
            elif extension == '.tsv':
                delimiter = '\t'


            df = pd.read_csv(os.path.join(self.feature_dir, fname), delimiter=delimiter)
            df['word'] = df['word'].str.lower()
            df = df[df['word'].notna()].reset_index() 

            features = {ft: [] for ft in feature_list}
            try:
                filtered_df = df[df['word'].isin(self.wordlist)]
                for word in self.wordlist:
                    row = filtered_df[filtered_df['word'] == word]
                    if len(row) == 0:
                        ## give 0 as default value
                        for ft in feature_list:
                            features[ft] += [0]
                    else:
                        for ft in feature_list:
                            features[ft] += [row[ft].values[0]]
                all_features.update(features)
            except KeyboardInterrupt:
                sys.exit()
            
        return all_features

    
    def make_dataframe(self):
        feat_df = pd.DataFrame()
        feat_df["word"] = self.wordlist

        for group, features in self.features_to_use.items():
            for cname, cfeat in self.feature_methods[group](features).items():
                feat_df[cname] = cfeat
        
        return feat_df
         


if __name__ == "__main__":
    with open('../data/wordlist/original_list_us_int.json', 'r') as f:
        org_wordlist = json.load(f)
        org_wordlist = org_wordlist.keys()

    df = pd.read_csv('../data/wordlist/noad_list.tsv', delimiter='\t')
    df['word'] = df['word'].astype(str)
    wordlist = [w.lower() for w in df['word']]
    wordlist = [w for w in wordlist if w not in org_wordlist] # needs prediction

    feature_extractor = FeatureExtractor(wordlist, FEATURE_DIR, FEATURES_TO_USE)
    feature_df = feature_extractor.make_dataframe()

    output_fname = 'noad'
    feature_df.to_csv(os.path.join('../data/train', f'{output_fname}_features.csv'), index=False)