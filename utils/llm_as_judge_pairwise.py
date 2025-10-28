from apis_config import openai_api_key
from openai import OpenAI
import json
import pandas as pd
from transformers import pipeline

def llm_as_judge_gpt(df, dest_path):    
    client = OpenAI(api_key=openai_api_key())
  
    c = 0    
    for question, answer1, answer2 in zip(df['Question'], df['Response A'], df['Response B']):
        c+=1
        prompt = f"Task: given this question, {question}, \
        please evaluate which is a better response between {answer1} and {answer2}. \
        Please return the response as either 'Response A' or 'Response B'."
        completion = {}
        response = client.responses.create(
        model="gpt-4o",
        input=prompt)
        completion[c] = response.output_text
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')


def llm_as_judge_gemma(df, dest_path):    
    pipe = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    #device="cuda",
    #torch_dtype=torch.bfloat16
)
    c = 0    
    for question, answer1, answer2 in zip(df['Question'], df['Response A'], df['Response B']):
        c+=1
        prompt =f"Task: given this question, {question}, \
        please evaluate which is a better response between {answer1} and {answer2}. \
        Please return the response as either 'Response A' or 'Response B'."
        completion = {}
        #response = client.chat.complete(
        #model = "mistral-large-latest",

        #messages = [{"role": "user", "content": prompt}]
        outputs = pipe(text=prompt, max_new_tokens=100)
        
        completion[c] = outputs[0]["generated_text"]#[-1]["content"]
        with open(dest_path, 'a') as f:
            f.write(json.dumps([completion]) + '\n')
 
df = pd.read_excel("data/pairwise_dataset.xlsx")
#llm_as_judge_gpt(df, "data/ll_as_judge_pairwise_gpt4o.jsonl")
llm_as_judge_gemma(df, "data/ll_as_judge_pairwise_gemma.jsonl")

