import glob
import pickle
import subprocess
import pandas as pd

set = 'test'
# set = 'validation'

for result_file in glob.glob('./results/*.txt'):
    print("Processing" + result_file)
    subprocess.call(
        'trec_eval -m all_trec -q ap_88_89/qrel_' + set + ' '
        + result_file + ' | grep -E "(^ndcg_cut_10)|(^P_5)|(^recall_1000)|(^map_cut_1000)" > ./trec_results_' + set
        + '/' + result_file.split('/')[-1], shell=True)

results = []

for trec_result_file in glob.glob('./trec_results_' + set + '/*.txt'):
    with open(trec_result_file, 'r') as f:
        df = pd.read_csv(trec_result_file, header=None, delimiter=r"\s+")
        df = df.pivot(index=1, columns=0, values=2)
        df = df[['ndcg_cut_10', 'map_cut_1000', 'P_5', 'recall_1000']]
        model_name = trec_result_file.split('_')[-1][0:-4] + '_'
        df.columns = [model_name + col_name for col_name in df.columns]
        results.append(df.copy())

results = pd.concat(results, axis=1)
print(results)
results.to_csv('results_' + set + '_task2.csv')
