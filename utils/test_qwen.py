from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the Qwen3-Reranker-0.6B model and tokenizer
# Make sure you have enough VRAM for the 8B model.
# For local deployment, consider quantized versions if VRAM is an issue.
# You might need to install 'accelerate' and 'bitsandbytes' for quantization.
model_name = "Qwen/Qwen3-Reranker-4B" # Using the corrected model name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
if torch.cuda.is_available():
    model.to("cuda")

def get_rerank_score(query: str, document: str) -> float:
    """
    Gets the relevance score from Qwen3-Reranker-8B for a query-document pair.
    """
    inputs = tokenizer(query, document, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # The reranker typically outputs a single logit representing relevance
        # You might need to apply a sigmoid to get a probability-like score if desired
        # CORRECTED LINE: Access the relevant logit (usually index 1 for binary classification)
        score = outputs.logits[0][1].item() # Access the score for the "positive" class
        # You can optionally apply a sigmoid to get a probability between 0 and 1
        # score = torch.sigmoid(outputs.logits[0][1]).item()
    return score

# Example Texts to Rank (from your previous code)
texts_to_rank = [
    {"id": "doc1", "content": "The capital of France is Paris. It's known for its beautiful architecture and vibrant culture. This text is very accurate and easy to read."},
    {"id": "doc2", "content": "Paris, the sprawling metropolis of France, often regarded as the epicenter of global fashion and art, exhibits unparalleled urban planning, albeit with occasional congestion issues."},
    {"id": "doc3", "content": "France's capital city, also Paris, has many good things. It is good for looking at buildings. The food is also good. This document has some errors."},
    {"id": "doc4", "content": "The capital of Germany is Berlin. This sentence is highly accurate and concise."}
]

# --- Strategy 1: Single Pass (Overall Quality) ---
print("--- Strategy 1: Single Pass (Overall Quality) ---")
overall_quality_query = "Please rank the following texts based on their overall quality, emphasizing factual accuracy and clear, concise readability."
ranked_by_overall_quality = []
for text_info in texts_to_rank:
    score = get_rerank_score(overall_quality_query, text_info["content"])
    ranked_by_overall_quality.append({"id": text_info["id"], "score": score, "content": text_info["content"]})

ranked_by_overall_quality.sort(key=lambda x: x["score"], reverse=True)
for item in ranked_by_overall_quality:
    print(f"ID: {item['id']}, Score: {item['score']:.4f}, Content: {item['content'][:100]}...")


print("\n--- Strategy 2: Multi-Pass and Combine Scores ---")
# Weights for combining scores (tune these based on your priorities)
weights = {
    "accuracy": 0.5,
    "readability": 0.3,
    "overall_quality_general": 0.2 # A general "overall quality" if not covered by accuracy/readability
}

multi_parameter_scores = {}
for text_info in texts_to_rank:
    doc_id = text_info["id"]
    content = text_info["content"]
    multi_parameter_scores[doc_id] = {"content": content}

    # Score for Accuracy
    accuracy_query = "Please evaluate the factual accuracy of this text. A higher score indicates greater accuracy and verifiable information."
    multi_parameter_scores[doc_id]["accuracy_score"] = get_rerank_score(accuracy_query, content)

    # Score for Readability
    readability_query = "Please evaluate the readability of this text. A higher score indicates clearer, more concise, and easier-to-understand language."
    multi_parameter_scores[doc_id]["readability_score"] = get_rerank_score(readability_query, content)

    # Score for General Overall Quality (as a fallback or general relevance)
    general_quality_query = "Please assess the overall quality and relevance of this document."
    multi_parameter_scores[doc_id]["general_quality_score"] = get_rerank_score(general_quality_query, content)

# Calculate final combined scores
final_ranked_texts = []
for doc_id, scores in multi_parameter_scores.items():
    combined_score = (
        weights["accuracy"] * scores["accuracy_score"] +
        weights["readability"] * scores["readability_score"] +
        weights["overall_quality_general"] * scores["general_quality_score"]
    )
    final_ranked_texts.append({
        "id": doc_id,
        "combined_score": combined_score,
        "accuracy_score": scores["accuracy_score"],
        "readability_score": scores["readability_score"],
        "general_quality_score": scores["general_quality_score"],
        "content": scores["content"]
    })

final_ranked_texts.sort(key=lambda x: x["combined_score"], reverse=True)

print("\nFinal Rankings (Multi-Pass and Combined):")
for item in final_ranked_texts:
    print(f"ID: {item['id']}, Combined Score: {item['combined_score']:.4f}")
    print(f"  - Accuracy: {item['accuracy_score']:.4f}")
    print(f"  - Readability: {item['readability_score']:.4f}")
    print(f"  - General Quality: {item['general_quality_score']:.4f}")
    print(f"  - Content: {item['content'][:100]}...\n")
