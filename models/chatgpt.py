from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm
from apis_config import openai_api_key
import time

client = OpenAI(api_key=openai_api_key())

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):
    for question in tqdm(df["question"]):
        completion = {}
        # Make a simple request
        response = client.responses.create(
        model="gpt-4o",
        input=question)
        completion[question] = response.output_text
        # wait 20 seconds
        time.sleep(15)
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "data/dataset_curated.xlsx"
output_path = "data/chatgpt/dataset_completion_chatgpt_4o.jsonl"
return_responses(load_data(data_path), output_path)