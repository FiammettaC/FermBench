import json
import re
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

def extract_scores(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # JSONL can sometimes be a list of dicts per line or one dict per line
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                # If the line is a list, iterate; if it's a dict, make it a list of one
                items = entry if isinstance(entry, list) else [entry]
                
                for item in items:
                    for question, markdown_str in item.items():
                        # Extract JSON from Markdown fences
                        json_match = re.search(r'```json\n(.*?)\n```', markdown_str, re.DOTALL)
                        if json_match:
                            raw_scores = json.loads(json_match.group(1))
                            # NORMALIZATION: Lowercase keys and strip spaces
                            scores = {k.lower().replace(" ", "_").strip(): v for k, v in raw_scores.items()}
                            scores['question'] = question.strip()
                            results.append(scores)
            except Exception as e:
                print(f"Skipping malformed line in {file_path}: {e}")
                continue
    return pd.DataFrame(results)


models = {
    'ChatGPT': 'llm_as_judge_gpt_chatgpt.jsonl',
    'Claude': 'llm_as_judge_gpt_claude.jsonl',
    'Gemini': 'llm_as_judge_gpt_gemini.jsonl',
    'DeepSeek': 'llm_as_judge_gpt_deepseek.jsonl',
    'le Chat': 'llm_as_judge_gpt_lechat.jsonl'
}

path = 'data/llm_as_judge_gpt/'

# 2. Process and Align
all_dfs = {name: extract_scores(path+filename) for name, filename in models.items()}


# Define the metrics you want to test
metrics = ['factual_accuracy', 'comprehensiveness', 'coherence', 
           'concision', 'readability', 'helpfulness', 
           'insightfulness', 'overall_quality']

import scikit_posthocs as sp

# --- Statistical Analysis ---
for metric in metrics:
    # Align questions across all models
    metric_df = pd.DataFrame({
        name: df.set_index('question')[metric] for name, df in all_dfs.items()
    }).dropna()

    if not metric_df.empty:
        # 1. Friedman Test
        try:
            stat, p = friedmanchisquare(*[metric_df[col] for col in metric_df.columns])
            print(f"\nMETRIC: {metric.upper()}")
            print(f"Friedman p-value: {p:.4e}")

            if p < 0.05:
                # 2. Post-hoc Nemenyi (Specifically for Friedman)
                # This function expects the 'Wide' format we already have
                posthoc = sp.posthoc_nemenyi_friedman(metric_df)
                
                print("Significant differences (p < 0.05) found between pairs:")
                # We filter the matrix to only show values < 0.05
                sig_matrix = posthoc[posthoc < 0.05]
                # Drop rows/cols that are entirely NaN (not significant) to clean the output
                clean_output = sig_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
                
                if clean_output.empty:
                    print("Overall difference is significant, but no specific pair reached p < 0.05.")
                else:
                    print(clean_output)
            else:
                print("No significant difference between models for this metric.")
                
        except ValueError as e:
            # This handles the 'nan' / zero variance issue
            print(f"\nMETRIC: {metric.upper()}")
            print(f"Could not calculate: {e} (Check if all scores are identical)")
    else:
        print(f"No overlapping data found for metric: {metric}")