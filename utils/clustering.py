# %%

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from data_loading import load_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from random import shuffle

df = load_data('../data/dataset_merged_all.xlsx')

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
vectors_chatgpt = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_chatgpt']]
vectors_claude = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_claude']]

#def dimentionality_reduction(X):
#    pca = PCA(n_components=2)
#    return pca.fit_transform(X)

#X_red_g = dimentionality_reduction(vectors_gemini)
#X_red_d = dimentionality_reduction(vectors_deepseek)
#X_red_l = dimentionality_reduction(vectors_lechat)

def dimentionality_reduction(X1, X2, X3, X4, X5):
    pca = PCA(n_components=2)
    X = X1 + X2 + X3 + X4 + X5
    return pca.fit_transform(X)

X_red = dimentionality_reduction(vectors_gemini, vectors_deepseek, vectors_lechat, vectors_chatgpt, vectors_claude)
X_red_gm = X_red[:118]
X_red_ds = X_red[118:236]
X_red_lc = X_red[236:354]
X_red_gpt = X_red[354:472]
X_red_cl = X_red[472:]

plt.figure(figsize=(10, 6), dpi=80)
plt.scatter(X_red_gm[:20, 0], X_red_gm[:20, 1], c=df['color'].iloc[:int], marker=".", alpha=1, s=100)
plt.scatter(X_red_ds[:20, 0], X_red_ds[:20, 1], c=df['color'].iloc[:int], marker="+", alpha=1, s=100)
plt.scatter(X_red_lc[:20, 0], X_red_lc[:20, 1], c=df['color'].iloc[:int], marker='*', alpha=1, s=100)
plt.scatter(X_red_gpt[:20, 0], X_red_gpt[:20, 1], c=df['color'].iloc[:int], marker="X", alpha=1, s=100)
plt.scatter(X_red_cl[:20, 0], X_red_cl[:20, 1], c=df['color'].iloc[:int], marker='h', alpha=1, s=100)
plt.grid(True)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.title(f'First two principal components in {int} sentences compared across models')
plt.show()


# %%
def plot_values(int):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(X_red_gm[:int, 0], X_red_gm[:int, 1], c=df['color'].iloc[:int], marker=".", alpha=1, s=100)
    plt.scatter(X_red_ds[:int, 0], X_red_ds[:int, 1], c=df['color'].iloc[:int], marker="+", alpha=1, s=100)
    plt.scatter(X_red_lc[:int, 0], X_red_lc[:int, 1], c=df['color'].iloc[:int], marker='*', alpha=1, s=100)
    plt.scatter(X_red_gpt[:int, 0], X_red_gpt[:int, 1], c=df['color'].iloc[:int], marker="X", alpha=1, s=100)
    plt.scatter(X_red_cl[:int, 0], X_red_cl[:int, 1], c=df['color'].iloc[:int], marker='h', alpha=1, s=100)
    plt.grid(True)
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.title(f'First two principal components in {int} sentences compared across models')
    plt.show()
    

plot_values(2)
plot_values(20)

def plot_values2(int):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(X_red_gm[:int, 0], X_red_gm[:int, 1], c=cmap_list[:int], marker=".", alpha=1, s=100)
    plt.scatter(X_red_ds[:int, 0], X_red_ds[:int, 1], c=cmap_list[:int], marker="+", alpha=1, s=100)
    plt.scatter(X_red_lc[:int, 0], X_red_lc[:int, 1], c=cmap_list[:int], marker='*', alpha=1, s=100)
    plt.scatter(X_red_gpt[:int, 0], X_red_gpt[:int, 1], c=cmap_list[:int], marker="X", alpha=1, s=100)
    plt.scatter(X_red_cl[:int, 0], X_red_cl[:int, 1], c=cmap_list[:int], marker='h', alpha=1, s=100)
    plt.grid(True)
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.title(f'First two principal components in {int} sentences compared across models')
    plt.show()
    

plot_values2(2)
plot_values2(20)

# %%
"""
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
"""