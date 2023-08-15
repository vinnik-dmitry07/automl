from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/mtsamples.csv')
df = df.groupby('transcription').first().reset_index()
df = df[~df['transcription'].isna()]
df['medical_specialty'] = df['medical_specialty'].map(str.strip)
df['transcription'] = df['transcription'].map(str.lower)

labels = sorted(df['medical_specialty'].unique().tolist())
label_ids = df['medical_specialty'].map(labels.index).tolist()

np.savez('data/data_text.npz', x=df['transcription'], y=label_ids)

counter = Counter(df['medical_specialty'])
entries = np.array(list(counter.values()))
plt.hist(entries, bins='auto')
plt.show()
plt.hist(entries[entries < 40])
plt.show()

transcription_words = set(' '.join(df['transcription']).split())
keywords = ', '.join(df['keywords'].dropna()).lower().split(', ')
keywords = list(map(lambda w: w.rstrip(',').rstrip('.'), keywords))
keywords = set(filter(lambda k: 0 < len(k) < 20 and k in transcription_words, keywords))
keywords = sorted(list(keywords))

print(1, len(labels), sorted(list(counter.items()), key=lambda c: c[1]))
print(2, df.isna().sum(axis=0))
print(3, len(df), df['transcription'].nunique(), df['description'].nunique())
print(4, len(keywords))

matrix = np.zeros((len(df), len(keywords)), dtype=np.float32)
for i, transcription in enumerate(tqdm(df['transcription'])):
    for j, keyword in enumerate(keywords):
        if keyword in transcription:
            matrix[i, j] = 1.

np.savez('data/data_matrix.npz', x=matrix, y=label_ids)
