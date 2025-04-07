import torch 
import pandas as pd
import json 

data = []
with open('dataset_completion_deepseek.jsonl', 'r') as f:
    for line in f:
        json_obj = json.loads(line)
        for key, value in json_obj[0].items():
            data.append({'key': key, 'value': value})

df = pd.DataFrame(data)
df.to_excel('dataset_completion_deepseek.xlsx', index=False)