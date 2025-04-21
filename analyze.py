import json
import pandas as pd
from collections import Counter


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        rows = []
    for item in data:
        for model in ["with_rag", "with_out_rag"]:
            metrics = item[model]
            rows.append({
                "question": item["question"],
                "model": "RAG" if model == "with_rag" else "Non-RAG",
                "accuracy": metrics["fact_accuracy"],
                "relevance": metrics["relevance"],
                "completeness": metrics["completeness"],
                "fluency": metrics["fluency"]
        })
        df = pd.DataFrame(rows)
    return df

def basic_metrics(df):
    stats = df.groupby('model').agg({
        'accuracy': ['mean', 'var'],
        'relevance': ['mean', 'var'],
        'completeness': ['mean', 'var'],
        'fluency': ['mean', 'var']
    }).round(2)
    print(stats)

def distribution(df):
    for col in ['accuracy', 'relevance', 'completeness', 'fluency']:
        for model in ['RAG', 'Non-RAG']:
            print(f"\n{model} {col} distribution")
            subset = df[df['model'] == model][col]
            dist = Counter(subset)
            total = len(subset)
            print({k: f"{v} ({v/total:.1%})" for k, v in dist.items()})

if __name__ == '__main__':
    df = read_json("result.json")
    basic_metrics(df)
    distribution(df)
