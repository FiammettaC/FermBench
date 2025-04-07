# Import and use the library
import anthropic
import os
import pandas as pd

# Set up your API key
client = anthropic.Anthropic(
    api_key="your-api-key-here",  # Or use os.environ["ANTHROPIC_API_KEY"]
)

df = pd.read_excel("dataset_curated.xlsx")

completion = []
for question in df["question"]:
    print(question)
 
    # Make a simple request
    message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1000,
    messages=[{"role": "user", "content": question}])
    
    # Print the response
    completion.append(message.content)


df_completion = pd.DataFrame()
df_completion["id"] = range(len(completion))
df_completion["question"] = df["question"]
df_completion["completion"] = completion
df_completion.to_excel("dataset_completion_claude.xlsx", index=False)