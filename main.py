import os
import sys
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
from IDS import IDS
from rename_columns import renaming_dict
from model_evaluation import evaluate_model
from reduce_dataset import map_dict
warnings.filterwarnings("ignore")


np.random.seed(1234)
PATH = os.environ.get("CICIDS_PROJECT")
if not os.path.isdir(os.path.join(PATH, "Results")):
    os.mkdir(os.path.join(PATH, "Results"))
# train-test data proportions
TRAIN_PROP = 0.60
TEST_PROP  = 0.40


##########################################
# visualization of activity distribution #
# before removing any data               #
##########################################

df = pd.read_pickle("ids_df_formatted.pickle")

# get name of each network activity type
x_coords = df.label.value_counts().index
# get frequency of each network activity type
y_coords = df.label.value_counts().values

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.barh(x_coords, y_coords)

# add text to end of each bar
for index, value in enumerate(y_coords):
    ax.text(value + 1000, index, format(value, ","), color='black')
plt.title("Distribution of Network Activity in CICIDS2017 Dataset", fontsize=18)
plt.xlabel("Number of Instances", fontsize=14)
plt.ylabel("Network Activity Label", fontsize=14)
# adjust x-tick limits
plt.xticks(list(range(0, 3000000, 500000)))
# format x-tick labels with commas for readability
ax.set_xticklabels(["{:,}".format(x) for x in ax.get_xticks()])
plt.show()

#################
# preprocessing #
#################

def port_category_encoding(x):
    '''Function to bin port number feature.'''
    if int(x) in list(range(0, 1024)):
        return '0:1023'
    elif int(x) in list(range(1024, 49152)):
        return '1024:49151'
    else:
        return '49152:65535'

df = pd.read_pickle("ids_df_formatted_and_reduced.pickle")

# one hot encode ports according to categories
df['port_category'] = df['destination_port'].apply(port_category_encoding)
for port_category in df['port_category'].unique():
    col_name = 'port_category_' + port_category
    df[col_name] = df['port_category'].apply(lambda x: 1 if x == port_category else 0)
    df[col_name] = df[col_name].astype('int')
df.drop(['destination_port', 'port_category'], axis=1, inplace=True)

# normalize the data to range [0,1]
min_max_scaler = {}
for col in df.select_dtypes(include=['int','float']).columns:
    minn = df[col].min()
    maxx = df[col].max()
    min_max_scaler[col] = {"min": minn, "max": maxx}
    if minn != maxx:
        df[col] = df[col].apply(lambda x: (x-minn)/(maxx-minn))
    else:
        df[col] = df[col].apply(lambda x: 0)
# pickle for use on new data
with open("min_max_scaler.pickle", "wb") as pickle_out:
    pickle.dump(min_max_scaler, pickle_out)

def partition_dataFrame(df):
    benign_indexes = df[df.label == "BENIGN"].index
    malicious_indexes = df[df.label != "BENIGN"].index

    # number of benign examples > number of malicious examples
    # balance proportion of benign & malicious data for training
    if len(benign_indexes) > len(malicious_indexes):
        extra_indexes = np.random.choice(benign_indexes, size=(len(benign_indexes)-len(malicious_indexes)), replace=False)
        extra_benign_data = df.loc[extra_indexes, df.columns != "label"].copy()
        extra_benign_labels = df.loc[extra_indexes, 'label'].copy()
        indexes_to_retain = list(set(list(df.index)) - set(extra_indexes))
    else:
        extra_benign_data = None
        extra_benign_labels = None
        indexes_to_retain = df.index

    data = df.loc[indexes_to_retain, df.columns != "label"].copy()
    labels = df.loc[indexes_to_retain, 'label'].copy()
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        train_size=TRAIN_PROP, test_size=TEST_PROP,
                                                        random_state=0, stratify=labels)
    # add extra benign instances to test data
    x_test = x_test.append(extra_benign_data)
    y_test = y_test.append(extra_benign_labels)
    l1 = set(list(x_train.index))
    l2 = set(list(x_test.index))
    # ensure no observation overlap between sets
    assert(len(l1.intersection(l2)) == 0)

    # benign/normal activity: 0, malicious activity: 1
    global label_conversion
    label_conversion = lambda x: 0 if x == 'BENIGN' else 1
    y_train = y_train.apply(label_conversion)
    y_test = y_test.apply(label_conversion)

    return (x_train, x_test, y_train, y_test)

x_train, x_test, y_train, y_test = partition_dataFrame(df)
data = df.loc[:, df.columns != "label"]
labels = df.loc[:, "label"]

###################################
# random forest feature selection #
###################################

feature_sets = {}
cols = list(x_train.columns)
title = "Random Forest Feature Selection"
print("\n{}\n".format(title) + "-"*len(title))
while cols:
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rf.fit(x_train.loc[:, cols].values, y_train.values)
    accuracy = rf.score(x_test.loc[:, cols].values, y_test.values)
    print("Number of Features:", len(cols), "->", accuracy)
    feature_sets[tuple(cols)] = accuracy
    temp_dict = dict(zip(cols, rf.feature_importances_))
    cols = sorted(temp_dict, key=lambda x: temp_dict[x])
    del cols[0]

# plot the change in test accuracy
X = []
Y = []
for key in sorted(feature_sets, key=lambda x: feature_sets[x]):
    X.append(len(key))
    Y.append(feature_sets[key])
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.barh(X, Y, color="lavender", edgecolor="black")
ax.set_facecolor("whitesmoke")
ax.set_xlim(0.96, 1.0)
ax.set_ylim(min(X)-1, max(X)+1)
plt.yticks(fontsize=9)
plt.title("Change in Test Accuracy in Response to Feature Set Reduction", fontsize=20)
plt.xlabel("Accuracy", fontsize=20)
plt.ylabel("Number of Features Retained", fontsize=20)
plt.show()
    
    
# plot the distribution of benign VS malicious using most important RF feature
b_index = labels[labels == "BENIGN"].index
m_index = labels[labels != "BENIGN"].index

plt.figure(figsize=(16,10))
plt.hist(df.loc[b_index, "pkt_len_std"], label="Benign", bins=30, alpha=0.5)
plt.hist(df.loc[m_index, "pkt_len_std"], label="Malicious", bins=30, alpha=0.5)
plt.title("Distribution of Packet Length Standard Deviation", fontsize=25)
plt.xlabel("Packet Length Standard Deviation", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.legend(fontsize=20)
plt.show()

# plot the distribution of benign VS individual attack types using most important RF feature
plt.figure(figsize=(16,10))
for label in labels.unique():
    temp_index = labels[labels == label].index
    plt.hist(df.loc[temp_index, "pkt_len_std"], label=label, bins=30, alpha=0.5)
plt.title("Distribution of Packet Length Standard Deviation", fontsize=25)
plt.xlabel("Packet Length Standard Deviation", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.legend(fontsize=20)
plt.show()

# columns to retain determined by examining RF feature selection results
cols_to_retain = ['fwd_seg_size_avg', 'max_pkt_len', 'flow_iat_mean',
                  'subflow_fwd_bytes', 'flow_iat_max', 'init_win_bytes_fwd',
                  'total_bwd_pkts', 'fwd_pkt_len_max', 'avg_pkt_size',
                  'pkt_len_variance', 'bwd_seg_size_avg', 'bwd_header_len',
                  'pkt_len_std', 'bwd_pkt_len_mean', 'pkt_len_mean',
                  'bwd_pkt_len_std', 'init_win_bytes_bwd']

##################
## HYPOTHESIS 1 ##
##################

def parallelized_search(matches_to_search, b_data, m_data, b_indexes, m_indexes, mp_list):
    indexes = {"benign": [], "malicious": []}
    for match in matches_to_search:
        benign = [i for i, v in enumerate(b_data) if v == match]
        malicious = [i for i, v in enumerate(m_data) if v == match]
        for i in benign:
            indexes["benign"].append(b_indexes[i])
        for i in malicious:
            indexes["malicious"].append(m_indexes[i])
    mp_list.append(indexes)

def search_for_matches(df, columns_to_search):
    '''Function to compare benign and malicious instances to determine the presence of matches.
    Uses multiprocessing to distribute the search space. Returns a dictionary containing 1) the unique
    values that are found by identifying the intersection of the set of benign values and the set
    of malicious values 2) the indexes where each unique value is found.'''
    cpu_count = os.cpu_count()
    benign_indexes = df[df.label == "BENIGN"].index
    malicious_indexes = df[df.label != "BENIGN"].index

    # determine if there are any benign instances that are identical to any malicious instances
    benign_data = [tuple(df.loc[index, columns_to_search].values) for index in benign_indexes]
    malicious_data = [tuple(df.loc[index, columns_to_search].values) for index in malicious_indexes]
    matches = set(benign_data) & set(malicious_data)
    match_indexes = {"benign": [], "malicious": []}
    with mp.Manager() as manager:
        queue = manager.list()
        processes = []
        segments = np.linspace(start=0, stop=len(matches), num=cpu_count+1).astype("int")
        for i in range(cpu_count):
            print('\rSearching through segment {} of {}'.format(i+1, cpu_count), end='')
            sys.stdout.flush()
            start, stop = segments[i], segments[i+1]
            processes.append(mp.Process(target=parallelized_search,
                                        args=(list(matches)[start:stop], benign_data, malicious_data,
                                              benign_indexes, malicious_indexes, queue)))
        [p.start() for p in processes]
        [p.join() for p in processes]
        for dictionary in queue:
            for key in dictionary:
                match_indexes[key].extend(dictionary[key])

    print("\n\nBenign-Malicious Matches\n" + "-"*25 + "\n{}".format(df.loc[match_indexes["malicious"], "label"].value_counts()))
    # free up memory
    del benign_data, malicious_data
    
    return {"values": matches, "indexes": match_indexes}


temp_df = pd.read_pickle("ids_df_formatted_and_reduced.pickle")
################################################
# determine matches before modifying feature set
################################################
results = search_for_matches(temp_df, temp_df.columns[temp_df.columns != "label"])
with open("identical_data-unmodified_feature_set.pickle", "wb") as pickle_out:
    pickle.dump(results, pickle_out)

################################################
# determine matches after modifying feature set
# i.e. reduced according to RF feature selection
################################################
results = search_for_matches(temp_df, cols_to_retain)
with open("identical_data-modified_feature_set.pickle", "wb") as pickle_out:
    pickle.dump(results, pickle_out)
del temp_df  

##################
## HYPOTHESIS 2 ##
##################

# indexes of matches determined after RF feature selection that will be dropped from subsequent analysis
indexes_to_exclude = results["indexes"]["benign"] + results["indexes"]["malicious"]
indexes_to_retain = list(set(list(df.index)) - set(indexes_to_exclude))
# exclude match examples and reduce the feature set using the columns that have been decided on
df_copy = df.loc[indexes_to_retain, cols_to_retain + ["label"]].copy()
x_train, x_test, y_train, y_test = partition_dataFrame(df_copy)
data = df_copy.loc[:, df_copy.columns != "label"].copy()
labels = df_copy.loc[:, "label"].copy()

########################################
# preliminary investigation            #
# logistic regression vs random forest #
########################################
log_mdl = LogisticRegression(solver='lbfgs', max_iter=10000, random_state=0)
log_mdl.fit(x_train.values, y_train.values.ravel())
log_mdl_preds = log_mdl.predict(x_test.values)
title = "Logistic Regression on Test Data"
print("\n\n{}\n".format(title) + "-"*len(title))
_ = evaluate_model(log_mdl_preds, y_test)
print("\n" + "="*70 + "\n")

rf_mdl = RandomForestClassifier(n_estimators=10, random_state=0)
rf_mdl.fit(x_train.values, y_train.values.ravel())
rf_mdl_preds = rf_mdl.predict(x_test.values)
title = "Random Forest on Test Data"
print("\n{}\n".format(title) + "-"*len(title))
_ = evaluate_model(rf_mdl_preds, y_test)
print("\n" + "="*70 + "\n")


######################################################
# PART 1:                                            #
# experiment with different stacked model parameters #
######################################################
TRAIN_SIZE = 100000
TEST_SIZE = 50000

x_train_temp = x_train.iloc[0:TRAIN_SIZE, :].copy()
y_train_temp = y_train.iloc[0:TRAIN_SIZE].copy()
x_test_temp = x_test.iloc[0:TEST_SIZE, :].copy()
y_test_temp = y_test.iloc[0:TEST_SIZE].copy()

records = pd.DataFrame(columns=["Primary Threshold", "Secondary Threshold", "Distance Threshold",
                                "Accuracy", "Precision", "Recall", "False Neg Rate", "False Pos Rate"])


rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(x_train_temp.values, y_train_temp.values.ravel())
rf_probs = rf.predict_proba(x_test_temp.values)
accum = 0
title = "Experiment with Stacked RF Model Parameters"
print("\n{}\n".format(title) + "-"*len(title))
for i in [0.80, 0.90, 1.0]:
    for j in [0.51, 0.75, 1.0]:
        ids = IDS(primary_benign_threshold=i, secondary_benign_threshold=j, rand_state=0, display_progress=False)
        ids.train(x_train_temp, y_train_temp)
        for k in [0.0, 0.001]:
            print("Primary Benign Threshold  : {:.2f}".format(i))
            print("Secondary Benign Threshold: {:.2f}".format(j))
            print("Closeness Threshold       : {:.3f}".format(k))
            ids_preds = ids.predict(x_test_temp, closeness_threshold=k)
            ids_results = evaluate_model(ids_preds, y_test_temp)
            records.loc[accum, :] = [i, j, k, ids_results["accuracy"],
                                     ids_results["precision"], ids_results["recall"],
                                     len(ids_results["indexes"]["false_neg"])/TEST_SIZE,
                                     len(ids_results["indexes"]["false_pos"])/TEST_SIZE]
            accum += 1
            print("\n" + "="*70 + "\n")
    
    # control group for each primary benign threshold
    rf_preds = [0 if prob[0] >= i else 1 for prob in rf_probs]
    rf_results = evaluate_model(rf_preds, y_test_temp)
    records.loc[accum, :] = [i, np.nan, np.nan, rf_results["accuracy"],
                             rf_results["precision"], rf_results["recall"],
                             len(rf_results["indexes"]["false_neg"])/TEST_SIZE,
                             len(rf_results["indexes"]["false_pos"])/TEST_SIZE]
    accum += 1
    print("\n" + "="*70 + "\n")

records.to_csv(os.path.join(PATH, "Results", "hypothesis_2-part_1.csv"), index=False, header=True, mode='w')

##########################################
# PART 2:                                #
# test updating feature of stacked model #
##########################################
PRIMARY_BENIGN_THRESHOLD = 1.0
SECONDARY_BENIGN_THRESHOLD = 0.95
first_cycle = True
result_cols = ["Model", "Update", "Distance Threshold", "Accuracy", "Precision", "Recall", "FN Rate", "FP Rate"]
rf_result_history = {"Hard_Vote": pd.DataFrame(columns=result_cols),
                     "Soft_Vote": pd.DataFrame(columns=result_cols),
                     "Stacked"  : pd.DataFrame(columns=result_cols)}

rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(x_train_temp.values, y_train_temp.values)
title = "Test of Update Feature of Stacked RF Model"
print("\n{}\n".format(title) + "-"*len(title))
for update in [True, False]:
    for threshold in [0.00001, 0.01]:
        rf_stack = IDS(num_estimators=50, primary_benign_threshold=PRIMARY_BENIGN_THRESHOLD,
                       secondary_benign_threshold=SECONDARY_BENIGN_THRESHOLD,
                       rand_state=0, display_progress=False)
        rf_stack.train(x_train_temp, y_train_temp)
        # split test data into 4 folds
        segments = np.linspace(0, y_test_temp.shape[0], 5).astype("int")
        for index in range(len(segments)-1):
            start, stop = segments[index], segments[index+1]
            X = x_test_temp.iloc[start:stop, :]
            Y = y_test_temp.iloc[start:stop]
            # two parameters being tested do not apply here so only need to determine these results once
            if first_cycle:
                print("\nRandom Forest - Hard Vote")
                r1_preds = rf.predict(X.values)
                r1 = evaluate_model(r1_preds, Y)
                print("\n" + "="*70 + "\n")
                
                print("\nRandom Forest - Soft Vote")
                r2_probs = rf.predict_proba(X.values)
                r2_preds = [0 if p[0] >= PRIMARY_BENIGN_THRESHOLD else 1 for p in r2_probs]
                r2 = evaluate_model(r2_preds, Y)
                print("\n" + "="*70 + "\n")

            rf_stack_preds = rf_stack.predict(X, closeness_threshold=threshold)
            print("\nRandom Forest Stack Model")
            r3 = evaluate_model(rf_stack_preds, Y)
            for model in {True : [(r1, "Hard_Vote"), (r2, "Soft_Vote"), (r3, "Stacked")],
                          False: [(r3, "Stacked")]}[first_cycle]:
                model_results = model[0]
                model_name = model[1]
                if model_name == "Hard_Vote" or model_name == "Soft_Vote":
                    u, t = np.nan, np.nan
                else:
                    u, t = update, threshold
                values_to_append = [model_name, u, t, model_results["accuracy"],
                                    model_results["precision"], model_results["recall"],
                                    len(model_results["indexes"]["false_neg"])/X.shape[0],
                                    len(model_results["indexes"]["false_pos"])/X.shape[0]]
                values_to_append = dict(zip(result_cols, values_to_append))
                rf_result_history[model_name] = rf_result_history[model_name].append(values_to_append,
                                                                                     ignore_index=True)
            # simulate a human analyst who updates the model according to false positive determinations
            if update:
                fp = r3["indexes"]["false_pos"]
                rf_stack.update_model(benign_misclassification_updates=[tuple(entry) for entry in x_test_temp.loc[fp, :].values])
            print("\n" + "="*70 + "\n")
            
        first_cycle = False

complete_results = pd.DataFrame(columns=result_cols)
for key in rf_result_history:
    complete_results = complete_results.append(rf_result_history[key])
complete_results.to_csv(os.path.join(PATH, "Results", "hypothesis_2-part_2.csv"), index=False, header=True, mode='w')


##################
## HYPOTHESIS 3 ##
##################

def reformat(path):
    '''Function to format csv files that have come from the CICFlowMeter.'''
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()
    df.rename(columns=renaming_dict, inplace=True)
    # the following two columns are read in by Pandas as type "object" but should be type "float"
    df['flow_bytes/s'] = df['flow_bytes/s'].astype('float64')
    df['flow_pkts/s'] = df['flow_pkts/s'].astype('float64')
    labels = df.loc[:, "label"]
    df = df.loc[:, cols_to_retain]
    for col in df.columns:
        # get average of non-nan values for imputing missing values
        avg = np.nanmean(df[col].values)
        df[col].fillna(avg, inplace=True)
        vals = list(filter(lambda x: x >= 0, df[col].values))
        if len(vals) > 0:
            min_val = sorted(vals)[0]
            max_val = list(filter(lambda x: x not in [np.inf, float('inf')], df[col].values))[-1]
            # impute values < 0
            df[col] = df[col].apply(lambda x: min_val if x < 0 else x)
            # impute infinity values
            df[col] = df[col].map(map_dict({float('inf'): max_val}))

    return (df, labels)


# get the external data/labels
external_benign_data, external_benign_labels = reformat(os.path.join(PATH, "External_Data",
                                                                     "external_benign.csv"))
external_malicious_data, external_malicious_labels = reformat(os.path.join(PATH, "External_Data",
                                                                           "external_malicious.csv"))
external_data = external_benign_data.append(external_malicious_data)
external_labels = external_benign_labels.append(external_malicious_labels)
external_data.reset_index(drop=True, inplace=True)
external_labels.reset_index(drop=True, inplace=True)

# isolate the indexes of examples with a class label featured in the external dataset
cic_indexes = {}
select_labels = labels[
    (labels == "Bot") | (labels == "Heartbleed") | (labels == "DoS GoldenEye") | (labels == "BENIGN")]
for index in select_labels.index:
    label = labels.loc[index]
    if label not in cic_indexes.keys():
        cic_indexes[label] = [index]
    else:
        cic_indexes[label].append(index)

# split into train & test sets based on class label
cic_train = {}
cic_test = {}
for key in cic_indexes.keys():
    cic_test[key] = list(np.random.choice(cic_indexes[key],
                                          size=int(np.ceil(0.40 * len(cic_indexes[key]))),
                                          replace=False))
    cic_train[key] = list(set(cic_indexes[key]) - set(cic_test[key]))

# for test sets -> determine the minimum amount of examples per class label between CICIDS2017 & external datasets
label_count = {}
for key in cic_test.keys():
    label_count[key] = min(len(cic_test[key]), external_labels.value_counts()[key])

cic_train_indexes = []
cic_test_indexes = []
for key in cic_train:
    cic_train_indexes.extend(cic_train[key])
    cic_test_indexes.extend(cic_test[key])
cic_test_data = data.loc[cic_test_indexes, :]
cic_test_labels = labels.loc[cic_test_indexes]
other_malicious_indexes = list(
    labels[(labels != "Bot") & (labels != "Heartbleed") & (labels != "DoS GoldenEye") & (labels != "BENIGN")].index)

# data & labels for training a model on a reduced dataset
reduced_df = data.loc[cic_train_indexes, :]
reduced_labels = labels.loc[cic_train_indexes].apply(label_conversion)
# vs total set of labels
complete_df = data.loc[cic_train_indexes + other_malicious_indexes, :]
complete_labels = labels.loc[cic_train_indexes + other_malicious_indexes].apply(label_conversion)
# train random forest using reduced data
rf_reduced = RandomForestClassifier(n_estimators=50, random_state=0)
rf_reduced.fit(reduced_df.values, reduced_labels.values.ravel())
# train random forest using non-reduced data
rf_complete = RandomForestClassifier(n_estimators=50, random_state=0)
rf_complete.fit(complete_df.values, complete_labels.values.ravel())

accum_results = {"cic": {"model 1": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0},
                         "model 2": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}},
                 "external": {"model 1": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0},
                              "model 2": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}}}

title = "Test of Generalizability"
print("\n{}\n\n".format(title) + "-" * len(title))
NUM_ITERATIONS = 50
for i in range(NUM_ITERATIONS):
    temp_indexes_cic = []
    temp_indexes_external = []
    for key in external_labels.value_counts().keys():
        temp_indexes_cic.extend(list(np.random.choice(cic_test_labels[cic_test_labels == key].index,
                                                      label_count[key], replace=False)))
        temp_indexes_external.extend(list(np.random.choice(external_labels[external_labels == key].index,
                                                           label_count[key], replace=False)))
    converted_labels_cic = labels.loc[temp_indexes_cic].apply(label_conversion)
    converted_labels_external = external_labels.loc[temp_indexes_external].apply(label_conversion)
    for dataset in accum_results:
        print("\n" + "=" * 30)
        print("Control") if dataset == "cic" else print('Test')
        print("=" * 30)
        info = {"cic": [data, converted_labels_cic, temp_indexes_cic],
                "external": [external_data, converted_labels_external, temp_indexes_external]}[dataset]
        title = "Complete RF"
        print("\n{}\n".format(title) + "-" * len(title))
        rf_complete_preds = rf_complete.predict(info[0].loc[info[2], :].values)
        rf_complete_indexes = evaluate_model(rf_complete_preds, info[1])
        title = "Reduced RF"
        print("\n\n{}\n".format(title) + "-" * len(title))
        rf_reduced_preds = rf_reduced.predict(info[0].loc[info[2], :].values)
        rf_reduced_indexes = evaluate_model(rf_reduced_preds, info[1])

        for model in accum_results[dataset]:
            rf_indexes = {"model 1": rf_complete_indexes, "model 2": rf_reduced_indexes}[model]
            for key in accum_results[dataset][model]:
                accum_results[dataset][model][key] = accum_results[dataset][model][key] + rf_indexes[key]

for dataset in accum_results:
    for model in accum_results[dataset]:
        for key in accum_results[dataset][model]:
            accum_results[dataset][model][key] = accum_results[dataset][model][key] / NUM_ITERATIONS