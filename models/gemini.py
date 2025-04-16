
from google import genai
import pandas as pd
import json
from config_apis import gemini_key
client = genai.Client(api_key=gemini_key())

def load_data(path):
    df = pd.read_excel(path)
    return df

def return_responses(df, dest_path):
    for question in df["question"]:
        completion = {}
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=question)
        completion[question] = response.text
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "../data/dataset_curated.xlsx"
output_path = "../data/dataset_completion_gemini_rerunX.jsonl"
return_responses(load_data(data_path), output_path)