# %%
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from data_loading import load_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import umap
import random

def dimentionality_reduction(X, method="pca"):
    if method == "pca":
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
    elif method == "tsne":
        tsne = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3)
        X_transformed = tsne.fit_transform(X)
    elif method == "umap":
        reducer = umap.UMAP()
        X_transformed = reducer.fit_transform(X)
    else:
        print("Please specify the correct method. Possible methods currently implemented: 'pca' or 'tsne'")
        X_transformed = 0
    return X_transformed

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
    plt.title(f'Dimensionality reduction of {(int2-int1)} responses compared across models')
    plt.show()

# Run code
df = load_data('../data/dataset_merged_all.xlsx')
df['color'] = random.sample(list(mcolors.CSS4_COLORS.values()), len(df))

#load sentence transformer to vectorize responses
model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#vectorize responses
vectors_gemini = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_gemini']]
vectors_deepseek = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_deepseek']]
vectors_lechat = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_lechat']]
vectors_chatgpt = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_chatgpt']]
vectors_claude = [model_sim.encode(resp, convert_to_tensor=False) for resp in df['value_claude']]

#dimensionality reduction
X_merged = vectors_gemini + vectors_deepseek + vectors_lechat + vectors_chatgpt + vectors_claude
X_red = dimentionality_reduction(np.array(X_merged))
X_red_gm = X_red[:117]
X_red_ds = X_red[117:234]
X_red_lc = X_red[234:351]
X_red_gpt = X_red[351:468]
X_red_cl = X_red[468:]

# plotting
plot_values(0, 5)
plot_values(0, 10)
plot_values(50, 70)
plot_values(0, 20)
plot_values(80, 113)
plot_values(0, 117)