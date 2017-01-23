import glob
import subprocess
import pandas as pd

for result_file in glob.glob('./results/*.txt'):
    subprocess.call(
        'trec_eval -m all_trec -q ap_88_89/qrel_test '
        + result_file + ' | grep -E "(^ndcg_cut_10)|(^P_5)|(^recall_1000)|(^map_cut_1000)" > ./trec_results/' +
        result_file.split('/')[-1], shell=True)

# for trec_result_file in glob.glob('./test/*.txt'):
#     with open(trec_result_file, 'r'):
#         pass