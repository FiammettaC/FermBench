from openai import OpenAI
import pandas as pd

client = OpenAI(api_key="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")

df = pd.read_excel("dataset_curated.xlsx")

completion = []
for question in df["question"]:
    print(question)
    # Make a simple request
    response = client.responses.create(
    model="gpt-3.5-turbo",
    input=question)
    completion.append(response)

df_completion = pd.DataFrame()
df_completion["id"] = range(len(completion))
df_completion["question"] = df["question"]
df_completion["completion"] = completion
df_completion.to_excel("dataset_completion_chatgpt.xlsx", index=False)