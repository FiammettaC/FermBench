import pandas as pd

df = pd.read_excel("questions_database.xlsx")
#print(df.head())
#print(df["Question 1"].tolist())

questions = []
for i in range(10):
    questions.extend(df[f"Question {i+1}"].tolist())

questions = [i for i in questions if str(i) != "nan"]

df_questions = pd.DataFrame()
df_questions["id"] = range(len(questions))
df_questions["question"] = questions
df_questions.to_excel("dataset_curated.xlsx", index=False)