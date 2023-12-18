import pandas as pd
import glob

score_dir = "score"
score_csvs = glob.glob(f"{score_dir}/*3.csv")

dfs = [pd.read_csv(csv) for csv in score_csvs]
df = pd.concat(dfs)

# without noise

df_without_noise = df[df["noise_class"] == "without_noise"]

# calculate accuracy
df_without_noise["correct"] = df_without_noise["predicted_label"] == df_without_noise["actual_label"]

print(f"\n🔥🔥🔥 -- score {'-'*100}\n")
print(f'without noise accuracy: {df_without_noise["correct"].sum() / len(df_without_noise)}')

# with noise ==========================================================================================================
df_with_noise = df[df["noise_class"] != "without_noise"]
df_with_noise = df_with_noise[df_with_noise["noise_class"] != df_with_noise["actual_label"]]

# calculate accuracy ---------------------------------------------------------------------------------------------------
df_with_noise["correct"] = df_with_noise["predicted_label"] == df_with_noise["actual_label"]
# confused model success rate -------------------------------------------------------------------------------------------
df_with_noise["confused_model_success"] = df_with_noise["predicted_label"] == df_with_noise["noise_class"]

print(f'with noise accuracy: {df_with_noise["correct"].sum() / len(df_with_noise):.2f}')
print(f'confused model success rate: {df_with_noise["confused_model_success"].sum() / len(df_with_noise):.2f}')

# calculate confused model success rate by noise class -------------------------------------------------------------------
df_with_noise_group = df_with_noise.groupby("noise_class")

for noise_class, _df in df_with_noise_group:
    print(f"{noise_class} confused model success rate: {_df['confused_model_success'].sum() / len(_df):.2f}")