from openai import OpenAI
import pandas as pd
import json

client = OpenAI(api_key="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")

df = pd.read_excel("../data/dataset_curated.xlsx")

#completion = []
for question in df["question"]:
    #print(question)
    completion = {}
    # Make a simple request
    response = client.responses.create(
    model="gpt-3.5-turbo",
    input=question)
    #completion.append(response)
    completion[question] = response
    with open("../data/dataset_completion_chatgpt.jsonl", 'a') as f:
        f.write(json.dumps([completion]) + '\n')

#df_completion = pd.DataFrame()
#df_completion["id"] = range(len(completion))
#df_completion["question"] = df["question"]
#df_completion["completion"] = completion
#df_completion.to_excel("../data/dataset_completion_chatgpt.xlsx", index=False)