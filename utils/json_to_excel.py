import torch 
import pandas as pd
import json 

def load_data_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            json_obj = json.loads(line)
            for key, value in json_obj[0].items():
                data.append({'key': key, 'value': value})
    return data

def save_excel(data, dest_path):
    df = pd.DataFrame(data)
    df.to_excel(dest_path, index=False)
    return df

# Run code
#jsonl_path = '../data/dataset_completion_gemini_subsample2.jsonl'
#output_path = '../data/dataset_completion_gemini_subsample2.xlsx'
#save_excel(load_data_jsonl(jsonl_path), output_path)

def excel_to_jsonl(path, dest_path):
    df = pd.read_excel(path)
    for k, v in zip(df['key'], df['value']):
        d =  {}
        d[k] = v
        with open(dest_path, 'a', encoding='utf-8-sig') as f:
            f.write(json.dumps([d]) + '\n')

#data_path = "../data/dataset_completion_lechat.xlsx"
#output_path = "../data/dataset_completion_lechat2.jsonl"
#excel_to_jsonl(data_path, output_path) 


#list_scores = load_data_jsonl("data/llm_as_judge_lechat_gemini.jsonl")
#print(list_scores)