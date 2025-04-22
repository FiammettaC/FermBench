from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch
import nltk
import time
import pandas as pd
from data_loading import load_answers

def cosine_similarity(prediction, truth):
    model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embedding_1 = model_sim.encode(prediction, convert_to_tensor=True)
    embedding_2 = model_sim.encode(truth, convert_to_tensor=True)
    # Similarity of two documents
    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)[0].cpu().numpy()
    return similarity[0]

def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def bleu_score(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([truth_tokens], pred_tokens, weights = (0.5, 0.5))
    return BLEUscore

def calculate_metrics(prediction, truth):
    similarity = cosine_similarity(prediction, truth)
    f1_score = compute_f1(prediction, truth)
    bleu = bleu_score(prediction, truth)
    return similarity, f1_score, bleu

def final_similarities(list_answers1, list_answers2):
    similarities_list = []
    f1_list = []
    bleu_list = []
    for i, j in zip(list_answers1, list_answers2):
        #start_time = time.time()
        similarities, f1_scores, bleu_scores = calculate_metrics(i, j)
        similarities_list.append(similarities)
        f1_list.append(f1_scores)
        bleu_list.append(bleu_scores)
        #print(f1_scores)
        #print(similarities)
        #print(bleu_scores)
        #print("--- %s seconds ---" % (time.time() - start_time))

    print(np.mean(np.array(similarities_list), axis=0))
    print(np.mean(np.array(f1_list), axis=0))
    print(np.mean(np.array(bleu_list), axis=0))

# Run code
gem_all = load_answers("gemini")
gem_all1 = load_answers("gemini_rerun1")
gem_all2 = load_answers("gemini_rerun2")

ds_all = load_answers("deepseek")
ds_all1 = load_answers("deepseek_rerun1")
ds_all2 = load_answers("deepseek_rerun2")

lc_all = load_answers("lechat")
lc_all1 = load_answers("lechat_rerun1")
lc_all2 = load_answers("lechat_rerun2")

gpt_all = load_answers("chatgpt")
gpt_all1 = load_answers("chatgpt_rerun1")
gpt_all2 = load_answers("chatgpt_rerun2")

cl_all = load_answers("claude")
cl_all1 = load_answers("claude_rerun1")
cl_all2 = load_answers("claude_rerun2")

print("Test 1: Gemini vs Gemini")
final_similarities(gem_all, gem_all1)
final_similarities(gem_all, gem_all2)
final_similarities(gem_all1, gem_all2)

print("Test 2: DeepSeek vs DeepSeek")
final_similarities(ds_all, ds_all1)
final_similarities(ds_all, ds_all2)
final_similarities(ds_all1, ds_all2)

print("Test 3: Le Chat vs Le Chat")
final_similarities(lc_all, lc_all1)
final_similarities(lc_all, lc_all2)
final_similarities(lc_all1, lc_all2)

print("Test 4: ChatGPT vs ChatGPT")
final_similarities(gpt_all, gpt_all1)
final_similarities(gpt_all, gpt_all2)
final_similarities(gpt_all1, gpt_all2)

print("Test 5: Claude vs Claude")
final_similarities(cl_all, cl_all1)
final_similarities(cl_all, cl_all2)
final_similarities(cl_all1, cl_all2)
