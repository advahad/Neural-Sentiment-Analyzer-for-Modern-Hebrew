import pandas as pd

FILE_PATH = '..\\results\\edge\\cnn\\unified_edge_data_07-09-2019_07-08-42.csv'
SAMPLE_SIZE = 60
NUMBER_OF_SAMPLES = 3
SAMPLES_DIR = '..\\results\\samples\\'

df = pd.read_csv(FILE_PATH, encoding='utf-8')
df = df[["edge_type","src_party","src_wing","dst_party","dst_wing","src_account",
                                             "dst_account","full_text"]]
df["human_label"] = ""
# df = df.drop(['preds_binary','preds_prob'])

all_samples = df.sample(SAMPLE_SIZE*NUMBER_OF_SAMPLES)
samples_list = [all_samples.iloc[SAMPLE_SIZE*i:SAMPLE_SIZE*i+SAMPLE_SIZE] for i in range(NUMBER_OF_SAMPLES)]
for i, sample in enumerate(samples_list):
    sample_df = pd.DataFrame(sample)
    sample_df.to_csv(SAMPLES_DIR + "sample_{}.csv".format(i), encoding='utf-8', header=True, index=True)
print("")