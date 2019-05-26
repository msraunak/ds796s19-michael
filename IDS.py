import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


class IDS(object):

    def __init__(self, num_estimators=50, primary_benign_threshold=1.0, secondary_benign_threshold=1.0,
                 rand_state=0, display_progress=True):
        self._display_progress = display_progress
        self._trained = False
        self._primary_benign_threshold = primary_benign_threshold
        self._secondary_benign_threshold = secondary_benign_threshold
        self._benign_misclassifications = []
        self._malicious_misclassifications = []
        self.primary_model = RandomForestClassifier(n_estimators=num_estimators, random_state=rand_state)
        self.secondary_model = RandomForestClassifier(n_estimators=num_estimators, random_state=rand_state)

    def _adjusted_predictions(self, probabilities, threshold_to_use):
        '''For a given probability array, predict 0 (i.e. "BENIGN") only if the probability of 0
        is greater than or equal to the specified threshold, else predict 1 (i.e. "MALICIOUS).'''
        threshold = {"primary": self._primary_benign_threshold,
                     "secondary": self._secondary_benign_threshold}[threshold_to_use]

        return [0 if prob[0] >= threshold else 1 for prob in probabilities]

    def _mock_prediction(self, data, labels, keyword):
        probabilities = self.primary_model.predict_proba(data.values)
        predictions = self._adjusted_predictions(probabilities, "primary")
        for index, prediction in enumerate(predictions):
            if self._display_progress and index % 1000 == 0:
                print('\rEvaluating {} data instance {} of {}'.format(keyword, index, len(predictions)), end='')
                sys.stdout.flush()
            instance_value = tuple(data.iloc[index, :].values)
            if prediction == 0 and instance_value not in self._benign_misclassifications and labels.iloc[index] != 0:
                self._benign_misclassifications.append(instance_value)
            elif prediction == 1:
                prob = self.secondary_model.predict_proba([data.iloc[index, :].values])
                pred = self._adjusted_predictions(prob, "secondary")
                if pred == 0 and instance_value not in self._benign_misclassifications and labels.iloc[index] != 0:
                    self._benign_misclassifications.append(instance_value)
                elif pred == 1 and instance_value not in self._malicious_misclassifications and labels.iloc[index] != 1:
                    self._malicious_misclassifications.append(instance_value)

    def train(self, training_data, training_labels, TRAIN_SIZE=0.80, CV_SIZE=0.20):
        training_data, cv_data, training_labels, cv_labels = train_test_split(training_data, training_labels,
                                                                              train_size=TRAIN_SIZE, test_size=CV_SIZE,
                                                                              random_state=0, stratify=training_labels)
        if self._display_progress: print("Training the primary model.")
        self.primary_model.fit(training_data.values, training_labels.values.ravel())
        probabilities = self.primary_model.predict_proba(training_data.values)
        predictions = self._adjusted_predictions(probabilities, "primary")

        results = self._evaluate_model(predictions, training_labels)
        for index in results['false_neg']:
            self._benign_misclassifications.append(tuple(training_data.loc[index, :].values))
        false_pos_indexes = results['false_pos']
        true_pos_indexes = results['true_pos']

        # create a dataset using the TP, FP data
        X = training_data.loc[false_pos_indexes, :]
        X = pd.concat([X, training_data.loc[true_pos_indexes, :]])
        Y = pd.Series(index=(false_pos_indexes + true_pos_indexes))
        Y.loc[true_pos_indexes] = 1
        Y.fillna(0, inplace=True)

        if self._display_progress: print("Training the secondary model.")
        self.secondary_model.fit(X.values, Y.values.ravel())
        self._mock_prediction(training_data, training_labels, "training")
        self._mock_prediction(cv_data, cv_labels, "cross-validation")
        if self._display_progress: print("\nDone training.")
        self._trained = True

    def _match_found_with_known_profile(self, predicted_value, entry, closeness_threshold):
        misclassification_profiles = {0: self._benign_misclassifications,
                                      1: self._malicious_misclassifications}
        if closeness_threshold == 0:
            return entry in misclassification_profiles[predicted_value]
        else:
            for known_profile in misclassification_profiles[predicted_value]:
                count = 0
                for known_val, entry_val in zip(known_profile, entry):
                    if known_val == entry_val:
                        count += 1
                    # if the distance between known profile value and entry value is greater than closeness threshold
                    elif abs(known_val - entry_val) > closeness_threshold:
                        break
                    else:
                        count += 1
                # the instance was similar (according to the closeness threshold) to the known profile for every feature
                if count == len(entry):
                    return True
            return False

    def predict(self, test_data, closeness_threshold=0):
        if not self._trained:
            raise NotFittedError(".train() must be called before .predict()")
        else:
            primary_probs = self.primary_model.predict_proba(test_data.values)
            primary_preds = self._adjusted_predictions(primary_probs, "primary")
            predictions = []
            for index, pred in enumerate(primary_preds):
                if self._display_progress and index % 1000 == 0:
                    print('\rEvaluating test data instance {} of {}'.format(index, len(primary_preds)), end='')
                    sys.stdout.flush()
                instance_value = tuple(test_data.iloc[index, :].values)
                match_found = self._match_found_with_known_profile(pred, instance_value, closeness_threshold)
                if pred == 0:
                    if not match_found:
                        # if instance_value not in self._benign_misclassifications:
                        predictions.append(0)
                    else:
                        predictions.append(1)
                else:
                    secondary_prob = self.secondary_model.predict_proba([test_data.iloc[index, :].values])
                    secondary_pred = self._adjusted_predictions(secondary_prob, "secondary")[0]
                    match_found = self._match_found_with_known_profile(secondary_pred, instance_value,
                                                                       closeness_threshold)
                    if secondary_pred == 0:
                        if not match_found:
                            # if instance_value not in self._benign_misclassifications:
                            predictions.append(0)
                        else:
                            predictions.append(1)

                    else:
                        if not match_found:
                            # if instance_value not in self._malicious_misclassifications:
                            predictions.append(1)
                        else:
                            predictions.append(0)

            return predictions

    def update_model(self, benign_misclassification_updates=None, malicious_misclassification_updates=None):
        if benign_misclassification_updates is not None:
            for entry in benign_misclassification_updates:
                self._benign_misclassifications.append(tuple(entry))
            self._benign_misclassifications = list(set(self._benign_misclassifications))
        if malicious_misclassification_updates is not None:
            for entry in malicious_misclassification_updates:
                self._malicious_misclassifications.append(tuple(entry))
            self._malicious_misclassifications = list(set(self._malicious_misclassifications))

    def _evaluate_model(self, predictions, labels):
        indexes = {'true_neg': [], 'true_pos': [],
                   'false_neg': [], 'false_pos': []}

        for pred, index in zip(predictions, labels.index):
            if pred == labels.loc[index]:
                if pred == 0:
                    indexes['true_neg'].append(index)
                else:
                    indexes['true_pos'].append(index)
            else:
                if pred == 0:
                    indexes['false_neg'].append(index)
                else:
                    indexes['false_pos'].append(index)

        return indexes