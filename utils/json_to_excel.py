import torch 
import pandas as pd
import json 

def load_data_jsonl(path):
    data = []
    with open(path, 'r') as f:
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