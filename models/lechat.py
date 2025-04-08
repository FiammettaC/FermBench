import os
from mistralai import Mistral
import pandas as pd
import json

api_key = "g2zTu5IGITxzDPA9TjSnjSS0GWCjQWjH"
#api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):
    for question in df["question"]:
        completion = {}
        chat_response = client.chat.complete(
        model = model,
        messages = [{"role": "user", "content": question}])
        completion[question] = chat_response.choices[0].message.content
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')


# Run code
data_path = "../data/dataset_curated.xlsx"
output_path = "../data/dataset_completion_lechat.jsonl"
return_responses(load_data(data_path), output_path)


