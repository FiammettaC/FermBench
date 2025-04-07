
from google import genai
import pandas as pd
import json

key = "AIzaSyA4APPbD1Y8IDclX-k9s75awBG2PK1BOxk"
client = genai.Client(api_key=key)

df = pd.read_excel("../data/dataset_curated.xlsx")

#completion = []
for question in df["question"]:
    completion = {}
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=question)
    #completion.append(response.text)
    completion[question] = response.text
    with open("../data/dataset_completion_gemini.jsonl", 'a') as f:
        f.write(json.dumps([completion]) + '\n')

#print(completion)

#df_completion = pd.DataFrame()
#df_completion["id"] = range(len(completion))
#df_completion["question"] = df["question"]
#df_completion["completion"] = completion
#df_completion.to_excel("../data/dataset_completion_gemini.xlsx", index=False)