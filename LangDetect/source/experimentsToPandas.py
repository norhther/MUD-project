import os
import pandas as pd

directory = '../output/'

data = []

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        clf = parts[0]
        kind = parts[1]
        vocab_size = int(parts[2])
        ngrams = parts[3].replace('[', '').replace(']', '').split(',')
        ngram_1 = int(ngrams[0])
        ngram_2 = int(ngrams[1].replace('.txt', ''))
        with open(os.path.join(directory, filename), 'r') as f:
            line = f.readline()
            coverage = float(line.split()[1])
            line = f.readline()
            f1_micro = float(line.split()[1])
            f1_macro = float(line.split()[3])
            f1_weighted = float(line.split()[5])
        data.append({'Classifier': clf,
                     'Kind': kind,
                     'Vocabulary Size': vocab_size,
                     'Ngram 1': ngram_1,
                     'Ngram 2': ngram_2,
                     'Coverage': coverage,
                     'F1 (micro)': f1_micro,
                     'F1 (macro)': f1_macro,
                     'F1 (weighted)': f1_weighted})
df = pd.DataFrame(data)
# sort by F1 (weighted) and Coverage
df = df.sort_values(by=['F1 (weighted)', 'Coverage'], ascending=False)
df.to_csv('../output/results.csv', index=False)
