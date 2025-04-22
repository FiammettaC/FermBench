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

print(len(list(mcolors.CSS4_COLORS.values())))

df['color'] = [np.random.choice(list(mcolors.CSS4_COLORS.values())) for i in range(len(df))] 
#load sentence transformer to vectorize responses
model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#vecrotize responses
vectors_gemini = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_gemini']]
vectors_deepseek = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_deepseek']]
vectors_lechat = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_lechat']]
vectors_chatgpt = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_chatgpt']]
vectors_claude = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_claude']]

def dimentionality_reduction2(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

def dimentionality_reduction(X1, X2, X3, X4, X5):
    pca = PCA(n_components=2)
    X = X1 + X2 + X3 + X4 + X5
    return pca.fit_transform(X)

X_red = dimentionality_reduction(vectors_gemini, vectors_deepseek, vectors_lechat, vectors_chatgpt, vectors_claude)
X_red_gm = X_red[:117]
X_red_ds = X_red[117:234]
X_red_lc = X_red[234:351]
X_red_gpt = X_red[351:468]
X_red_cl = X_red[468:]

print(len(vectors_gemini))
print(len(X_red_gm), len(X_red_ds), len(X_red_lc), 
      len(X_red_gpt), len(X_red_cl))

# %%
def plot_values(int1, int2):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.scatter(X_red_gm[int1:int2, 0], 
                X_red_gm[int1:int2, 1], 
                c=df['color'].iloc[int1:int2], 
                marker=".", alpha=1, s=100)
    plt.scatter(X_red_ds[int1:int2, 0], 
                X_red_ds[int1:int2, 1], 
                c=df['color'].iloc[int1:int2], 
                marker="+", alpha=1, s=100)
    plt.scatter(X_red_lc[int1:int2, 0], 
                X_red_lc[int1:int2, 1], 
                c=df['color'].iloc[int1:int2], 
                marker='*', alpha=1, s=100)
    plt.scatter(X_red_gpt[int1:int2, 0], 
                X_red_gpt[int1:int2, 1], 
                c=df['color'].iloc[int1:int2], 
                marker="X", alpha=1, s=100)
    plt.scatter(X_red_cl[int1:int2, 0], 
                X_red_cl[int1:int2, 1], 
                c=df['color'].iloc[int1:int2], 
                marker='h', alpha=1, s=100)
    plt.grid(True)
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.title(f'First two principal components in {int2} sentences compared across models')
    plt.show()
    

plot_values(0, 5)
plot_values(0, 10)
plot_values(50, 70)
plot_values(0, 20)
plot_values(80, 113)

X_red_gm = dimentionality_reduction2(vectors_gemini)
X_red_ds = dimentionality_reduction2(vectors_deepseek)
X_red_lc = dimentionality_reduction2(vectors_lechat)
X_red_gpt = dimentionality_reduction2(vectors_chatgpt)
X_red_cl = dimentionality_reduction2(vectors_claude)

plot_values(0, 5)
plot_values(0, 10)
plot_values(50, 70)
plot_values(0, 20)
plot_values(80, 113)