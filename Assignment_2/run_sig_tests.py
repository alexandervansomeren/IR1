import pandas as pd

df = pd.read_csv('results_task1.csv')
df.index = df['1']
df.drop('1', axis=1, inplace=True)
['ndcg_cut_10', 'map_cut_1000','P_cut_5', 'recall_cut_1000']
print df.describe()