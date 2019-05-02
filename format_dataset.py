import os
import pandas as pd


PATH = os.environ.get("CICIDS_PROJECT")

df = pd.read_csv(os.path.join(PATH, "CICIDS_Data", "ids_data.csv"))
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("forward", "fwd")
df.columns = df.columns.str.replace("backward", "bwd")
df.columns = df.columns.str.replace("length", "len")
df.columns = df.columns.str.replace("packet", "pkt")
df.columns = df.columns.str.replace("average", "avg")
df.rename(columns={"min_seg_size_fwd": "fwd_seg_size_min",
                   "avg_fwd_segment_size": "fwd_seg_size_avg",
                   "avg_bwd_segment_size": "bwd_seg_size_avg"}, inplace=True)
df["label"] = df["label"].str.replace(" �", "", regex=False)
df["label"] = df["label"].str.replace("slowloris", "Slow Loris", regex=False)
df["label"] = df["label"].str.replace("DoS Slowhttptest", 'DoS Slow HTTP Test', regex=False)
# the following two columns are read in by Pandas as type "object" but should be type "float"
df['flow_bytes/s'] = df['flow_bytes/s'].astype('float64')
df['flow_pkts/s'] = df['flow_pkts/s'].astype('float64')

# pickle dataframe for reuse
df.to_pickle("ids_df_formatted.pickle")