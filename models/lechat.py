import os
from mistralai import Mistral
import pandas as pd

api_key = "g2zTu5IGITxzDPA9TjSnjSS0GWCjQWjH"
#api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

df = pd.read_excel("../data/dataset_curated.xlsx")

completion = []
for question in df["question"]:
    #print(question)

    chat_response = client.chat.complete(
    model = model,
    messages = [{"role": "user", "content": question}])
    completion.append(chat_response.choices[0].message.content)

#print(completion)

df_completion = pd.DataFrame()
df_completion["id"] = range(len(completion))
df_completion["question"] = df["question"]
df_completion["completion"] = completion
df_completion.to_excel("../data/dataset_completion_lechat.xlsx", index=False)


