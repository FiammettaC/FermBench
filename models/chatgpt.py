from openai import OpenAI
import pandas as pd
import json

client = OpenAI(api_key="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")

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):
    for question in df["question"]:
        completion = {}
        # Make a simple request
        response = client.responses.create(
        model="gpt-3.5-turbo",
        input=question)
        completion[question] = response
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "../data/dataset_curated.xlsx"
output_path = "../data/dataset_completion_chatgpt.jsonl"
return_responses(load_data(data_path), output_path)