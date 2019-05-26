import os
import numpy as np
import pandas as pd

PATH = os.environ.get("CICIDS_PROJECT")

CICIDS_FILES = ['Monday-WorkingHours.pcap_ISCX.csv',
                'Tuesday-WorkingHours.pcap_ISCX.csv',
                'Wednesday-workingHours.pcap_ISCX.csv',
                'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv']

EXTERNAL_FILES = {"BENIGN": ['2013-10-21_capture-1-only-dns.pcap_Flow.csv',
                             '2013-12-17_capture1.pcap_Flow.csv',
                             '2015-03-24_capture1-only-dns.pcap_Flow.csv'],
                  "MALICIOUS": ['10.0.0.40.dradis.heartbleed.pcap_Flow.csv',
                                'botnet-capture-20110810-neris.pcap_Flow.csv',
                                'goldeneye.pcap_Flow.csv']}


def aggregate_csv_files(files, root_folder, sub_folder, combined_csv_name):
    # read in csv files into individual dataFrames
    print('\nReading csv files for', combined_csv_name)
    if root_folder == "CICIDS_Data":
        dfs = [pd.read_csv(os.path.join(PATH, root_folder, sub_folder, file)) for file in files]
    else:
        dfs = []
        for file in files:
            df = pd.read_csv(os.path.join(PATH, root_folder, sub_folder, file))
            if sub_folder == "Malicious":
                if file == '10.0.0.40.dradis.heartbleed.pcap_Flow.csv':
                    df["Label"] = np.repeat("Heartbleed", df.shape[0])
                elif file == 'botnet-capture-20110810-neris.pcap_Flow.csv':
                    df["Label"] = np.repeat("Bot", df.shape[0])
                else:
                    df["Label"] = np.repeat("DoS GoldenEye", df.shape[0])
            else:
                df["Label"] = np.repeat("BENIGN", df.shape[0])
            dfs.append(df)
    print('Files read.')

    # join dataFrames into one .csv file
    print('\nWriting dataFrame 1 of {} to csv file'.format(len(files)))
    dfs[0].to_csv(os.path.join(PATH, root_folder, combined_csv_name), index=False, header=True, mode='w')
    for index, df in enumerate(dfs[1:]):
        print('Writing dataFrame {} of {} to csv file'.format(index + 2, len(files)))
        df.to_csv(os.path.join(PATH, root_folder, combined_csv_name), index=False, header=False, mode='a')
    print('\nDone processing.')

if __name__ == "__main__":
    aggregate_csv_files(CICIDS_FILES, "CICIDS_Data", "original_csv_files", "ids_data.csv")
    aggregate_csv_files(EXTERNAL_FILES["BENIGN"], "External_Data", "Benign", "external_benign.csv")
    aggregate_csv_files(EXTERNAL_FILES["MALICIOUS"], "External_Data", "Malicious", "external_malicious.csv")