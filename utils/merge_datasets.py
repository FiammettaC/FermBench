from data_loading import load_question_answers
from functools import reduce
import pandas as pd

answers1 = load_question_answers("gemini").rename(columns={"value": "value_gemini"}).drop_duplicates(subset='key', keep='first')
answers2 = load_question_answers("deepseek").rename(columns={"value": "value_deepseek"}).drop_duplicates(subset='key', keep='first')
answers3 = load_question_answers("lechat").rename(columns={"value": "value_lechat"}).drop_duplicates(subset='key', keep='first')

data_frames = [answers1, answers2, answers3]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['key']), data_frames)

df_merged.to_excel('../data/dataset_merged_GDL.xlsx', index=False)