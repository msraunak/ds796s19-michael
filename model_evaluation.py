def evaluate_model(preds, labs):
    correct, incorrect = 0, 0
    indexes = {'true_neg': [], 'true_pos': [],
               'false_neg': [], 'false_pos': []}
    
    precision = lambda r: len(r["true_pos"])/(len(r["true_pos"]) + len(r["false_pos"]))
    recall = lambda r: len(r["true_pos"])/(len(r["true_pos"]) + len(r["false_neg"]))        
    
    for pred, index in zip(preds, labs.index):
        if pred == labs.loc[index]:
            correct += 1
            if pred == 0:
                indexes['true_neg'].append(index)
            else:
                indexes['true_pos'].append(index)
        else:
            incorrect += 1
            if pred == 0:
                indexes['false_neg'].append(index)
            else:
                indexes['false_pos'].append(index)
    
    a = correct/(correct+incorrect)
    try:
        p = precision(indexes)
    except:
        p = None
    try:
        r = recall(indexes)
    except:
        r = None
    
    print("\nAccuracy : {:.5f}".format(a))
    print("Precision: {:.5f}".format(p))
    print("Recall   : {:.5f}".format(r))
    print('\n\t\tPREDICTION: benign\tPREDICTION: malicious')
    print(str.expandtabs('\nACTUAL: benign\t{}\t{}'.format(len(indexes['true_neg']),
                                                           len(indexes['false_pos'])), 25))
    print(str.expandtabs('\nACTUAL: malicious\t{}\t{}'.format(len(indexes['false_neg']),
                                                              len(indexes['true_pos'])), 25))
    
    return {"indexes": indexes, "accuracy": a, "precision": p, "recall": r}

