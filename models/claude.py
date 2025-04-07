# Import and use the library
import anthropic
import os
import pandas as pd
import json

# Set up your API key
client = anthropic.Anthropic(
    api_key="your-api-key-here",  # Or use os.environ["ANTHROPIC_API_KEY"]
)

df = pd.read_excel("../data/dataset_curated.xlsx")

#completion = []
for question in df["question"]:
    completion = {}
    # Make a simple request
    message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1000,
    messages=[{"role": "user", "content": question}])
    
    # Print the response
    #completion.append(message.content)
    completion[question] = message.content
    with open("../data/dataset_completion_claude.jsonl", 'a') as f:
        f.write(json.dumps([completion]) + '\n')


#df_completion = pd.DataFrame()
#df_completion["id"] = range(len(completion))
#df_completion["question"] = df["question"]
#df_completion["completion"] = completion
#df_completion.to_excel("../data/dataset_completion_claude.xlsx", index=False)