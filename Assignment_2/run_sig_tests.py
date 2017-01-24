import pandas as pd

df = pd.read_csv('results_eval_task1.csv')
df.index = df['1']
df.drop('1', axis=1, inplace=True)
measure_names = ['ndcg_cut_10', 'map_cut_1000','P_cut_5', 'recall_cut_1000']
model_names = [c[0:-4] for c in df.columns if c[-3:] == 'P_5']
for measure_name in measure_names:
    pass
print(df.ix['all'])