from data_loading import load_question_answers
from functools import reduce
import pandas as pd
from json_to_excel import load_data_jsonl
import json

path = "data/"
path2 = "data/dataset_completion"

def merge_responses(path, extention="jsonl"):
    if extention=="jsonl":
        answers1 = pd.DataFrame(load_data_jsonl(f'{path}_gemini_rerun1.jsonl')).rename(columns={"value": "value_gemini"}).drop_duplicates(subset='key', keep='first')
        answers2 = pd.DataFrame(load_data_jsonl(f'{path}_deepseek_rerun1.jsonl')).rename(columns={"value": "value_deepseek"}).drop_duplicates(subset='key', keep='first')
        answers3 = pd.DataFrame(load_data_jsonl(f'{path}_lechat_rerun1.jsonl')).rename(columns={"value": "value_lechat"}).drop_duplicates(subset='key', keep='first')
        answers4 = pd.DataFrame(load_data_jsonl(f'{path}_chatgpt_rerun1.jsonl')).rename(columns={"value": "value_chatgpt"}).drop_duplicates(subset='key', keep='first')
        answers5 = pd.DataFrame(load_data_jsonl(f'{path}_claude_rerun1.jsonl')).rename(columns={"value": "value_claude"}).drop_duplicates(subset='key', keep='first')
        
    else:
        answers1 = load_question_answers("gemini").rename(columns={"value": "value_gemini"}).drop_duplicates(subset='key', keep='first')
        answers2 = load_question_answers("deepseek").rename(columns={"value": "value_deepseek"}).drop_duplicates(subset='key', keep='first')
        answers3 = load_question_answers("lechat").rename(columns={"value": "value_lechat"}).drop_duplicates(subset='key', keep='first')
        answers4 = load_question_answers("chatgpt").rename(columns={"value": "value_chatgpt"}).drop_duplicates(subset='key', keep='first')
        answers5 = load_question_answers("claude").rename(columns={"value": "value_claude"}).drop_duplicates(subset='key', keep='first')

    data_frames = [answers1, answers2, answers3, answers4, answers5]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['key']), data_frames)

    df_merged.to_excel('data/dataset_merged_all.xlsx', index=False)
    return df_merged


def merge_responses_all(path):
    models = ["gemini", "deepseek", "lechat", "chatgpt", "claude"]
    reruns =  ["", "_rerun1", "_rerun2"]
    answers = []
    for model in models:
        print(model)
        for rerun in reruns:
            print(rerun)
            try: 
                answers.append(pd.DataFrame(load_data_jsonl(f'{path}{model}/dataset_completion_{model}{rerun}.jsonl')).rename(columns={"value": f"value_{model}{rerun}"}).drop_duplicates(subset='key', keep='first'))
            except json.decoder.JSONDecodeError:
                print(model, rerun)
                continue
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['key']), answers)
    df_merged.to_excel('data/dataset_merged_all_reruns.xlsx', index=False)
    return df_merged
    
#merge_responses(path2, "jsonl")
#merge_responses_all(path)