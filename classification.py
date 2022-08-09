from sklearn.metrics import classification_report
import pandas as pd


def classif_lr(x, y, estimator, th):
    y_pred = estimator.predict_proba(x)[:, 1] > th
    print(classification_report(y, y_pred, labels=[1, 0]))


def classif(x, y, estimator):
    y_pred = estimator.predict(x)
    print(classification_report(y, y_pred, labels=[1, 0]))


def transform(data, label, scaler, num, cat):
    X = data.drop(label, axis=1)
    y = data[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    ohe = OneHotEncoder()

    X_train_transformed = pd.concat([pd.DataFrame(scaler.fit_transform(X_train[num]),
                                                  columns=scaler.feature_names_in_),
                                     pd.DataFrame(ohe.fit_transform(X_train[cat]).toarray(),
                                                  columns=ohe.get_feature_names_out(ohe.feature_names_in_))],
                                    axis=1)
    X_test_transformed = pd.concat([pd.DataFrame(scaler.transform(X_test[num]),
                                                 columns=scaler.feature_names_in_),
                                    pd.DataFrame(ohe.transform(X_test[cat]).toarray(),
                                                 columns=ohe.get_feature_names_out(ohe.feature_names_in_))],
                                   axis=1)

    return X_train_transformed, X_test_transformed, y_train, y_test


def perform(X, actual, est):
    TP = 0
    FN = 0
    TN = 0
    FP = 0

    actual = actual.reset_index(drop=True)
    predicted = est.predict(X)

    for i in range(len(actual)):
        if actual[i] == 0:
            if predicted[i] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if predicted[i] == 1:
                TP += 1
            else:
                FN += 1

    recall = TP/(TP+FN)

    TNR = TN/(TN+FP)

    if TP + FP != 0:
        precision = TP/(TP+FP)
    else:
        precision = 'none'

    if TN + FN != 0:
        NPV = TN/(TN+FN)
    else:
        NPV = 'none'

    acc = (TP+TN)/(len(actual))

    f1 = 2*recall*precision/(recall+precision)

    return recall, precision, TNR, NPV, f1, acc


def perf_df(X_train, y_train, X_test, y_test, model_list, label_list):
    perf_dict = {'recall': {},
                 'precision': {},
                 'TNR': {},
                 'NPV': {},
                 'accuracy': {},
                 'f1': {}}

    for x, y in zip(model_list, label_list):
        recall, precision, TNR, NPV, f1, acc = perform(X_train, y_train, x)
        suffix = '_train'
        perf_dict['recall'][y+suffix] = recall
        perf_dict['precision'][y+suffix] = precision
        perf_dict['TNR'][y+suffix] = TNR
        perf_dict['NPV'][y+suffix] = NPV
        perf_dict['accuracy'][y+suffix] = f1
        perf_dict['f1'][y+suffix] = acc

    for x, y in zip(model_list, label_list):
        recall, precision, TNR, NPV, f1, acc = perform(X_test, y_test, x)
        suffix = '_test'
        perf_dict['recall'][y+suffix] = recall
        perf_dict['precision'][y+suffix] = precision
        perf_dict['TNR'][y+suffix] = TNR
        perf_dict['NPV'][y+suffix] = NPV
        perf_dict['accuracy'][y+suffix] = f1
        perf_dict['f1'][y+suffix] = acc

    print(pd.DataFrame(perf_dict).sort_index())
