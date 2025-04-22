from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch
import nltk
import time
import pandas as pd
from data_loading import load_answers, load_data

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

def pairwise_similarities(list_answers1, list_answers2):
    cs_list = []
    f1_list = []
    bleu_list = []
    for i, j in zip(list_answers1, list_answers2):
        cos_sim, f1_scores, bleu_scores = calculate_metrics(i, j)
        cs_list.append(cos_sim)
        f1_list.append(f1_scores)
        bleu_list.append(bleu_scores)

    avg_cs = np.mean(np.array(cs_list), axis=0)
    avg_f1 = np.mean(np.array(f1_list), axis=0)
    avg_bleu = np.mean(np.array(bleu_list), axis=0)
    
    #print(avg_cs, avg_f1, avg_bleu)
    return avg_cs, avg_f1, avg_bleu

def average_similarities(df, model):
    avg_cs_x, avg_f1_x, avg_bleu_x = pairwise_similarities(df[f"value_{model}"], df[f"value_{model}_rerun1"])
    avg_cs_y, avg_f1_y, avg_bleu_y = pairwise_similarities(df[f"value_{model}"], df[f"value_{model}_rerun2"])
    avg_cs_z, avg_f1_z, avg_bleu_z = pairwise_similarities(df[f"value_{model}_rerun1"], df[f"value_{model}_rerun2"])
    
    avg_rerun_cs = (avg_cs_x + avg_cs_y + avg_cs_z)/3
    avg_rerun_f1 = (avg_f1_x + avg_f1_y + avg_f1_z)/3
    avg_rerun_bleu = (avg_bleu_x + avg_bleu_y + avg_bleu_z)/3
    
    print(round(avg_rerun_cs, 3), round(avg_rerun_f1, 3), round(avg_rerun_bleu, 3))
    return avg_rerun_cs, avg_rerun_f1, avg_rerun_bleu    

def average_similarities_different_models(df):
    models = ["gemini", "deepseek", "lechat", "chatgpt", "claude"]
    reruns =  ["", "_rerun1", "_rerun2"]
    for model1 in models[:-1]:
        for model2 in models[1:]:
            avg_cs_x, avg_f1_x, avg_bleu_x = pairwise_similarities(df[f"value_{model1}"], 
                                                                   df[f"value_{model2}"])
            avg_cs_y, avg_f1_y, avg_bleu_y = pairwise_similarities(df[f"value_{model1}_rerun1"], 
                                                                   df[f"value_{model2}_rerun1"])
            avg_cs_z, avg_f1_z, avg_bleu_z = pairwise_similarities(df[f"value_{model1}_rerun2"], 
                                                                   df[f"value_{model2}_rerun2"])
            print(model1, model2)
            avg_rerun_cs = (avg_cs_x + avg_cs_y + avg_cs_z)/3
            avg_rerun_f1 = (avg_f1_x + avg_f1_y + avg_f1_z)/3
            avg_rerun_bleu = (avg_bleu_x + avg_bleu_y + avg_bleu_z)/3
            
            print(round(avg_rerun_cs, 2), round(avg_rerun_f1, 2), round(avg_rerun_bleu, 2))
    return

# Run code
"""
df = load_data("data/dataset_merged_all_reruns.xlsx")

print("Test 1: Gemini vs Gemini")
average_similarities(df, "gemini")

print("Test 2: DeepSeek vs DeepSeek")
average_similarities(df, "deepseek")

print("Test 3: Le Chat vs Le Chat")
average_similarities(df, "lechat")

print("Test 4: ChatGPT vs ChatGPT")
average_similarities(df, "chatgpt")

print("Test 5: Claude vs Claude")
average_similarities(df, "claude")

print("##############")
average_similarities_different_models(df)
"""