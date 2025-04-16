# %%

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from data_loading import load_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from random import shuffle

df = load_data('../data/dataset_merged_GDL.xlsx')

import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
cmap = get_cmap(len(df))
cmap_list = [cmap(i) for i in range(len(df))]
shuffle(cmap_list)

df['color'] = [np.random.choice(list(mcolors.CSS4_COLORS.values())) for i in range(len(df))] 
#load sentence transformer to vectorize responses
model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#vecrotize responses
vectors_gemini = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_gemini']]
vectors_deepseek = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_deepseek']]
vectors_lechat = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_lechat']]

#def dimentionality_reduction(X):
#    pca = PCA(n_components=2)
#    return pca.fit_transform(X)

#X_red_g = dimentionality_reduction(vectors_gemini)
#X_red_d = dimentionality_reduction(vectors_deepseek)
#X_red_l = dimentionality_reduction(vectors_lechat)

def dimentionality_reduction(X1, X2, X3):
    pca = PCA(n_components=2)
    X = X1 + X2 + X3
    return pca.fit_transform(X)

X_red = dimentionality_reduction(vectors_gemini, vectors_deepseek, vectors_lechat)
X_red_g = X_red[:108]
X_red_d = X_red[108:216]
X_red_l = X_red[216:]

plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(X_red_g[:2, 0], X_red_g[:2, 1], c=df['color'].iloc[:2], marker="o", alpha=1, s=100)
plt.scatter(X_red_d[:2, 0], X_red_d[:2, 1], c=df['color'].iloc[:2], marker="+", alpha=1, s=100)
plt.scatter(X_red_l[:2, 0], X_red_l[:2, 1], c=df['color'].iloc[:2], marker='*', alpha=1, s=100)
plt.show()

# %%
plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(X_red_g[:20, 0], X_red_g[:20, 1], c=cmap_list[:20], marker="o", alpha=1, s=100)
plt.scatter(X_red_d[:20, 0], X_red_d[:20, 1], c=cmap_list[:20], marker="+", alpha=1, s=100)
plt.scatter(X_red_l[:20, 0], X_red_l[:20, 1], c=cmap_list[:20], marker='*', alpha=1, s=100)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(X_red_g[20:50, 0], X_red_g[20:50, 1], c=df['color'].iloc[20:50], marker="o", alpha=1, s=100)
plt.scatter(X_red_d[20:50, 0], X_red_d[20:50, 1], c=df['color'].iloc[20:50], marker="+", alpha=1, s=100)
plt.scatter(X_red_l[20:50, 0], X_red_l[20:50, 1], c=df['color'].iloc[20:50], marker='*', alpha=1, s=100)
plt.show()


plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(X_red_g[:, 0], X_red_g[:, 1], c=df['color'], marker="o", alpha=1, s=100)
plt.scatter(X_red_d[:, 0], X_red_d[:, 1], c=df['color'], marker="+", alpha=1, s=100)
plt.scatter(X_red_l[:, 0], X_red_l[:, 1], c=df['color'], marker='*', alpha=1, s=100)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(X_red_g[:, 0], X_red_g[:, 1], c='red', alpha=0.5)
plt.scatter(X_red_d[:, 0], X_red_d[:, 1], c='blue', alpha=0.5)
plt.scatter(X_red_l[:, 0], X_red_l[:, 1], c='orange', alpha=0.5)
plt.grid(False)
plt.xlabel('Model complexity --->')
plt.ylabel('Message length --->')
plt.title('Minimum Message Length')
plt.show()