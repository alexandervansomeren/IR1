import glob
import pickle
import subprocess
import pandas as pd

for result_file in glob.glob('./results/*.txt'):
    subprocess.call(
        'trec_eval -m all_trec -q ap_88_89/qrel_test '
        + result_file + ' | grep -E "(^ndcg_cut_10)|(^P_5)|(^recall_1000)|(^map_cut_1000)" > ./trec_results/' +
        result_file.split('/')[-1], shell=True)

results = []
for trec_result_file in glob.glob('./trec_results/*.txt'):
    with open(trec_result_file, 'r') as f:
        df = pd.read_csv(trec_result_file, header=None, delimiter=r"\s+")
        df = df.pivot(index=1, columns=0, values=2)
        model_name = trec_result_file.split('_')[-1][0:-4] + '_'
        df.columns = [model_name + col_name for col_name in df.columns]
        results.append(df.copy())

results = pd.concat(results, axis=1)
results.to_csv('results_task1.csv')
