import pandas as pd

df = pd.read_excel("dataset_merged_all_reruns - Copy.xlsx")

#print(df.head())

section = "Here, you will evaluate the first set of responses on a series of parameters.\n \
The parameters are:\n- Factual accuracy aims to assess to what extent the answer is correct.\n \
- Comprehensiveness measures the completeness of the response, whether it answers the question fully or if there is something missing.\n \
- Coherence focuses on whether the response tackles the question and aligns with user intent.\n - Concision evaluates whether the response is redundant or contains superfluous information.\n \
\n- Readability provides an overview of how easy the response is to read and understand or if it is too complex, if the language is fluent, natural and grammatically correct. \
\n- Helpfulness evaluates whether the answer is useful to the user/student. \
\n- Insightfulness tackles the depth of the answer and if it offers any novel perspectives or goes beyond a superficial answer. \
\n- Overall quality provides a general assessment of the response."

all_data = []
types = []
parameters = []
scale = []
for idx, row in df.iloc[30:].iterrows():

    all_data.append(section)
    all_data.append(row[0])
    all_data.append(row[1])
    all_data.append(row[2])
    all_data.append(row[3])
    all_data.append(row[4])
    all_data.append(row[5])
    all_data.append(row[6])
    
    types.append('Section')
    types.append('Paragraph')
    types.append('MULTIPLE CHOICE GRID')
    types.append('MULTIPLE CHOICE GRID')
    types.append('MULTIPLE CHOICE GRID')
    types.append('MULTIPLE CHOICE GRID')
    types.append('MULTIPLE CHOICE GRID')
    types.append('Paragraph')
    
    parameters.append(None)
    parameters.append(None)
    parameters.append('Factual accuracy, Comprehensiveness, Coherence, Concision, Readability, Helpfulness, Insightfulness, Overall quality')
    parameters.append('Factual accuracy, Comprehensiveness, Coherence, Concision, Readability, Helpfulness, Insightfulness, Overall quality')
    parameters.append('Factual accuracy, Comprehensiveness, Coherence, Concision, Readability, Helpfulness, Insightfulness, Overall quality')
    parameters.append('Factual accuracy, Comprehensiveness, Coherence, Concision, Readability, Helpfulness, Insightfulness, Overall quality')
    parameters.append('Factual accuracy, Comprehensiveness, Coherence, Concision, Readability, Helpfulness, Insightfulness, Overall quality')
    parameters.append(None)
    
    scale.append(None)
    scale.append(None)
    scale.append('1, 2, 3, 4, 5')
    scale.append('1, 2, 3, 4, 5')
    scale.append('1, 2, 3, 4, 5')
    scale.append('1, 2, 3, 4, 5')
    scale.append('1, 2, 3, 4, 5')
    scale.append(None)
    
#print(all_data)

df2 = pd.DataFrame()
df2['Question'] = all_data
df2['Types'] = types
df2['Parameters'] = parameters
df2['Scale'] = scale



print(df2)

df2.to_excel('to_google_form2.xlsx')
