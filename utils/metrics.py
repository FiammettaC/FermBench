from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch
import nltk
import time
import pandas as pd

def load_answers(dataset):
    df = pd.read_excel(f"../data/dataset_completion_{dataset}.xlsx")
    return df['value'].tolist()

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
answers1 = load_answers("gemini_subsample1")
answers2 = load_answers("gemini_subsample2")
answers3 = load_answers("deepseek")
answers3 = answers3[:(int(len(answers2)))]
answers4 = load_answers("lechat")
answers4 = answers4[:(int(len(answers2)))]

print("Test 1: Gemini vs Gemini")
final_similarities(answers1, answers2)

print("Test 2: Gemini vs DeepSeek")
final_similarities(answers1, answers3)

print("Test 3: Gemini vs Le Chat")
final_similarities(answers1, answers4)

print("Test 4: Le Chat vs DeepSeek")
final_similarities(answers4, answers3)