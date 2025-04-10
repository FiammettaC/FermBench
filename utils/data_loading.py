import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    return df

def load_answers(dataset):
    df = pd.read_excel(f"../data/dataset_completion_{dataset}.xlsx")
    return df['value'].tolist()

def load_question_answers(dataset):
    df = pd.read_excel(f"../data/dataset_completion_{dataset}.xlsx")
    return df