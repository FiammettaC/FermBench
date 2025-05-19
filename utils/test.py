from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    device="cpu",
    torch_dtype=torch.bfloat16, 
    use_fast=True
)

prompt = f"Given the following question 'How modeling the fermentation process can influence our life?', \
        please assess this answer 'the sun is yellow' based on these factors, \
        on a 1-5 scale (5 being the highest) for each factor: \
        factual accuracy, comprehensiveness, coherence, concision, \
        readability, helpfulness, insightfulness and overall quality. \
        Here is an explanations of the various factors.\
        Factual accuracy aims to assess to what extent the answer is correct \
        or if the model is hallucinating. \
        Comprehensiveness measures the completeness of the response, \
        whether it answers the question fully or if there is something missing.\
        Coherence focuses on whether the response tackles the question and aligns with user intent. \
        Concision evaluates whether the response is redundant or contains superfluous information. \
        Readability provides an overview of how easy the response is to read and understand \
        or if it is too complex, if the language is fluent, natural and grammatically correct. \
        Helpfulness evaluates whether the answer is useful to the user/student. \
        Insightfulness tackles the depth of the answer and if it offers any novel perspectives \
        or goes beyond a superficial answer. \
        Overall quality provides a general assessment of the response.\
        Please return your answer as a dictionary, where the key is the factor the score is the value. \
        Do not elaborate further."

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            #{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
# Okay, let's take a look! 
# Based on the image, the animal on the candy is a **turtle**. 
# You can see the shell shape and the head and legs.
