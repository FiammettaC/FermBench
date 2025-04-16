from openai import OpenAI
import pandas as pd
import json

#key = "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"
key = "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"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

def load_data(path):
    df = pd.read_excel(path)
    #print(df)
    return df

def return_responses(df, dest_path):
    for question in df["question"]:
        completion = {}
        chat_response = client.chat.completions.create(
        model="deepseek/deepseek-chat:free",
        messages=[{"role": "user", "content": question}])
        #print(chat_response.choices[0].message.content)
        completion[question] = chat_response.choices[0].message.content
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')

# Run code
data_path = "../data/dataset_curated.xlsx"
output_path = "../data/dataset_completion_deepseek_rerun1.jsonl"
return_responses(load_data(data_path), output_path)