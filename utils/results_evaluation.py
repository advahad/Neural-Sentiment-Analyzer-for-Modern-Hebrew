import pandas as pd
PATH = '../data/compare_samples.csv'
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

th_values = [0, 0.75, 0.85, 0.95]

df = pd.read_csv(PATH)
df = df[['verdict','Algorithm','prob']]
df = df[0:180]
df = df[df.verdict != 0]
print (df.shape)
# df.verdict = df.verdict.astype(int).apply(lambda x: 0 if x == -1 else 1)
# df.Algorithm = df.Algorithm.astype(int).apply(lambda x: 0 if x == -1 else 1)

df.verdict = df.verdict.astype(int)
df.Algorithm = df.Algorithm

# analysis_df = pd.DataFrame([], columns=['Threshold', 'Instances size', 'Accuracy'])
analysis_df = pd.DataFrame([], columns=['Threshold', 'Instances size', 'Accuracy', 'Roc Auc'])
for th in th_values:
    temp = df[df.prob > th]

    accuracy_res = accuracy_score(temp.verdict.values, temp.Algorithm.values)
    #
    # confusion_matrix = confusion_matrix(temp.verdict.values, temp.Algorithm.values)

    roc_auc_score_res = roc_auc_score(temp.verdict.values, temp.Algorithm.values)

    # f1_score_res = f1_score(temp.verdict.values, temp.Algorithm.values)
    to_append = [th, temp.shape[0], accuracy_res, roc_auc_score_res]
    analysis_df = analysis_df.append(pd.Series(to_append, index=analysis_df.columns), ignore_index=True)
    # print("{}".format(th))
    # print(temp.shape)
    # print(accuracy_res)
    # # print(confusion_matrix)
    # print(roc_auc_score)
    # print(f1_score_res)

analysis_df.to_csv("../results/samples_compare_analysis_no_neutral_without_neutral.csv",  index=False)