from openai import OpenAI
import pandas as pd
import json

key = "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"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

df = pd.read_excel("../data/dataset_curated.xlsx")

#completion = []
for question in df["question"]:
    completion = {}
    chat_response = client.chat.completions.create(
    model="deepseek/deepseek-chat:free",
    messages=[{"role": "user", "content": question}])
    #completion.append(chat_response.choices[0].message.content)
    completion[question] = chat_response.choices[0].message.content
    with open("../data/dataset_completion_deepseek.jsonl", 'a') as f:
        f.write(json.dumps([completion]) + '\n')

#df_completion = pd.DataFrame()
#df_completion["id"] = range(len(completion))
#df_completion["question"] = df["question"]
#df_completion["completion"] = completion
#df_completion.to_excel("../data/dataset_completion_deepseek.xlsx", index=False)