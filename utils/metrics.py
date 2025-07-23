from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
import torch
#import nltk
import time
import pandas as pd
from data_loading import load_answers, load_data
#from mistralai import Mistral
#from apis_config import lechat_key
import transformers
from transformers import pipeline, AutoProcessor, BitsAndBytesConfig #Llama4ForConditionalGeneration
from apis_config import openai_api_key
import time
from openai import OpenAI
import torchvision
from huggingface_hub import hf_hub_download


TF_ENABLE_ONEDNN_OPTS=0
torchvision.disable_beta_transforms_warning()
 
torch.cuda.empty_cache()
 
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
 
def average_words(df):
    models = ["gemini", "deepseek", "lechat", "chatgpt", "claude"]
    reruns =  ["", "_rerun1", "_rerun2"]
    
    for model in models:
        print(model)
        length = 0
        c = 0
        for run in reruns:
            print(run)
            for sentence in df[f"value_{model}{run}"]:
                c += 1
                length += len(sentence.split())
        print(f"Total for {model}: {round(length/c, 2)}")
        
"""      
def llm_as_judge(df, model, dest_path):
    client = Mistral(api_key=lechat_key())
    for question, answer in zip(df['key'], df[f'value_{model}']):
        prompt = f"Given the following question '{question}', \
        please assess this answer '{answer}' based on these factors, \
        on a 1-5 scale (5 being the highest) for each factor: \
        factual accuracy, comprehensiveness, coherence, concision, \
        readability, helpfulness, insightfulness and overall quality. \
        Here is an explanations of the various factors.\
        Factual accuracy aims to assess to what extent the answer is correct \
        or if the model is hallucinating. \
        Comprehensiveness measures the completeness of the response, \
        whether it answers the question fully or if there is something missing.\
        Coherence focuses on whether the response tackles the question and aligns with user intent. \
        Concision evaluates whether the response is redundant or contains superfluous information. \
        Readability provides an overview of how easy the response is to read and understand \
        or if it is too complex, if the language is fluent, natural and grammatically correct. \
        Helpfulness evaluates whether the answer is useful to the user/student. \
        Insightfulness tackles the depth of the answer and if it offers any novel perspectives \
        or goes beyond a superficial answer. \
        Overall quality provides a general assessment of the response.\
        Please return your answer as a dictionary, where the key is the factor the score is the value. \
        Do not elaborate further."
        completion = {}
        response = client.chat.complete(
        model = "mistral-large-latest",
        messages = [{"role": "user", "content": prompt}])
        completion[question] = response.choices[0].message.content
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')
            
"""
def llm_as_judge(df, model, dest_path):    
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        #torch_dtype=torch.bfloat16,
        device_map="auto",
    )
  
    for question, answer in zip(df['key'], df[f'value_{model}']):
        prompt = f"Given the following question '{question}', \
        please assess this answer '{answer}' based on these factors, \
        on a 1-5 scale (5 being the highest) for each factor: \
        factual accuracy, comprehensiveness, coherence, concision, \
        readability, helpfulness, insightfulness and overall quality. \
        Here is an explanations of the various factors.\
        Factual accuracy aims to assess to what extent the answer is correct \
        or if the model is hallucinating. \
        Comprehensiveness measures the completeness of the response, \
        whether it answers the question fully or if there is something missing.\
        Coherence focuses on whether the response tackles the question and aligns with user intent. \
        Concision evaluates whether the response is redundant or contains superfluous information. \
        Readability provides an overview of how easy the response is to read and understand \
        or if it is too complex, if the language is fluent, natural and grammatically correct. \
        Helpfulness evaluates whether the answer is useful to the user/student. \
        Insightfulness tackles the depth of the answer and if it offers any novel perspectives \
        or goes beyond a superficial answer. \
        Overall quality provides a general assessment of the response.\
        Please return your answer as a dictionary, where the key is the factor the score is the value. \
        Do not elaborate further."
        completion = {}
        #response = client.chat.complete(
        #model = "mistral-large-latest",
 
        messages = [{"role": "user", "content": prompt}]
        outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
        
        print(outputs)
        print(outputs[0]["generated_text"][-1])
        
        completion[question] = outputs[0]["generated_text"][-1]
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')
 
def llm_as_judge_gemma(df, model, dest_path):    
    pipe = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16)}
)
  
    for question, answer in zip(df['key'], df[f'value_{model}']):
        prompt = f"Task: given a specific question, please evaluate the given answer \
        on a 1-5 scale (5 being the highest) for each of these factors: \
        factual accuracy, comprehensiveness, coherence, concision, \
        readability, helpfulness, insightfulness and overall quality. \
        An explanations of the various factors is provided below.\
        Factual accuracy aims to assess to what extent the answer is correct \
        or if the model is hallucinating. \
        Comprehensiveness measures the completeness of the response, \
        whether it answers the question fully or if there is something missing.\
        Coherence focuses on whether the response tackles the question and aligns with user intent. \
        Concision evaluates whether the response is redundant or contains superfluous information. \
        Readability provides an overview of how easy the response is to read and understand \
        or if it is too complex, if the language is fluent, natural and grammatically correct. \
        Helpfulness evaluates whether the answer is useful to the user/student. \
        Insightfulness tackles the depth of the answer and if it offers any novel perspectives \
        or goes beyond a superficial answer. \
        Overall quality provides a general assessment of the response.\
        Please return your answer as a dictionary, where the key is the factor the score is the value. \
        Do not elaborate further. The question asked is: {question}, and its relative answer is: {answer}."
        completion = {}
 
        messages = [{"role": "user", "content": prompt}]
        with torch.no_grad():
            outputs = pipe(messages, max_new_tokens=200)
        
        completion[question] = outputs[0]["generated_text"][-1]["content"]
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')
        
def llm_as_judge_gpt(df, model, dest_path):    
    client = OpenAI(api_key=openai_api_key())
  
    for question, answer in zip(df['key'], df[f'value_{model}']):
        prompt = f"Task: given a specific question, please evaluate the given answer \
        on a 1-5 scale (5 being the highest) for each of these factors: \
        factual accuracy, comprehensiveness, coherence, concision, \
        readability, helpfulness, insightfulness and overall quality. \
        An explanations of the various factors is provided below.\
        Factual accuracy aims to assess to what extent the answer is correct \
        or if the model is hallucinating. \
        Comprehensiveness measures the completeness of the response, \
        whether it answers the question fully or if there is something missing.\
        Coherence focuses on whether the response tackles the question and aligns with user intent. \
        Concision evaluates whether the response is redundant or contains superfluous information. \
        Readability provides an overview of how easy the response is to read and understand \
        or if it is too complex, if the language is fluent, natural and grammatically correct. \
        Helpfulness evaluates whether the answer is useful to the user/student. \
        Insightfulness tackles the depth of the answer and if it offers any novel perspectives \
        or goes beyond a superficial answer. \
        Overall quality provides a general assessment of the response.\
        Please return your answer as a dictionary, where the key is the factor the score is the value. \
        Do not elaborate further. The question asked is: {question}, and its relative answer is: {answer}."
        completion = {}
        response = client.responses.create(
        model="gpt-4o",
        input=prompt)
        completion[question] = response.output_text
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')
 
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

# Run LLM as judge GPT
#df = load_data("data/dataset_merged_all_reruns.xlsx")
#m = "gemini"
#llm_as_judge_gemma(df, m, f"data/llm_as_judge_gemma_{m}.jsonl")


from json_to_excel import load_data_jsonl
import re

models = ['chatgpt', 'claude', 'gemini', 'deepseek', 'lechat'] #'chatgpt', 
runs = ['', '_rerun1', '_rerun2']
judge = 'gemma'

#dict_chatgpt = {}
#dict_claude = {}
#dict_lechat = {}
#dict_gemini = {}
#dict_deepseek = {}

for model in models:
    dict = {}
    fa = 0
    comp = 0
    coh = 0
    conc = 0
    read = 0
    help = 0
    ins = 0
    oq = 0
    missed = 0
    for run in runs:
        data = load_data_jsonl(f"data/llm_as_judge_{judge}/llm_as_judge_{judge}_{model}{run}.jsonl")
        for idx, d in enumerate(data):
            numbers = re.findall(r'[+-]?\d*\.?\d+', d['value'])
            #print(numbers)
            numbers = [round(float(n)) for n in numbers]
            #print(numbers)
            #print(len(numbers))
            if len(numbers) == 8:
                fa += int(numbers[0])
                comp += int(numbers[1])
                coh += int(numbers[2])
                conc += int(numbers[3])
                read += int(numbers[4])
                help += int(numbers[5])
                ins += int(numbers[6])
                oq += int(numbers[7])
            else:
                #print(model, run, d)
                missed +=1
                continue
        dict['factual_accuracy'] = round(fa/(len(data)*3), 2)
        dict['comprehensiveness'] = round(comp/(len(data)*3), 2)
        dict['coherence'] = round(coh/(len(data)*3), 2)
        dict['concision'] = round(conc/(len(data)*3), 2)
        dict['readability'] = round(read/(len(data)*3), 2)
        dict['helpfulness'] = round(help/(len(data)*3), 2)
        dict['insightfulness'] = round(ins/(len(data)*3), 2)
        dict['overall_quality'] = round(oq/(len(data)*3), 2)  
    print(model)
    print(dict) 
    print(missed) 
                

          
"""
json_string = data[0]['value'].strip('```json python {}')
print(json_string)


#print(d)
            #print(idx)
            #try: 
            #    json_string = d['value'].strip('```json python {}').lower()
            #    print(json_string)
                
            #    python_dict = json.loads(json_string)
            #except json.decoder.JSONDecodeError:
            #    try:
            #        print(d['value'].lower())
            #        python_dict = json.loads(d['value'].lower())
            #    except json.decoder.JSONDecodeError:
            #        python_dict = json.loads(d['value'].strip('```python').lower())
            
            
            #try:
            #    print(python_dict['factual_accuracy'])
            #    print(int(python_dict['factual_accuracy']))
                fa += int(python_dict['factual_accuracy'])
            except KeyError:
                fa += int(python_dict['factual accuracy'])
                
            comp += python_dict['comprehensiveness']
            coh += python_dict['coherence']
            conc += python_dict['concision']
            read += python_dict['readability']
            help += python_dict['helpfulness']
            ins += python_dict['insightfulness']
            try:
                oq += python_dict['overall_quality']
            except KeyError:
                oq += python_dict['overall quality']
"""  
