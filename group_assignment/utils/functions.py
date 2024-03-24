from sklearn import metrics
import matplotlib.pyplot as plt
import catboost as cb
import numpy as np
from sklearn.metrics import precision_recall_curve


def training(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

def predict(model, x_test, threshold = 0.5):
    
    proba = model.predict_proba(x_test)
    preds = np.where(proba[:,1] > threshold, 1, 0)
    return preds, proba

def get_metrics(y_pred, y_proba, y_test, plot=False):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
    specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
    cm = metrics.confusion_matrix(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_proba[:,1])
    if plot:
        disp = metrics.ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.show()
    else:
        pass
    final_metrics = {
        'accuracy': accuracy, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'auc': auc}
    return final_metrics, cm

def main_process(model, x_train, y_train, x_test, y_test, plot=False, threshold=0.5, roc_curve = False):
    model = training(model, x_train, y_train)
    y_pred, y_proba = predict(model, x_test, threshold)
    if roc_curve:
        plot_roc_auc(y_test, y_proba)
    else:
        pass
    metrics, cm = get_metrics(y_pred, y_proba, y_test, plot)
    return model, metrics, cm


def objective(trial, x_train, y_train, x_val, y_val):

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "depth": trial.suggest_int("depth", 1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 1000, 10000),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    model = cb.CatBoostClassifier(**param, random_state=123)

    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0, early_stopping_rounds=100)

    preds, proba = predict(model, x_val)
    metrics, cm = get_metrics(preds, proba, y_val, False)
    auc = metrics['auc']
    return auc


def plot_roc_auc(y_test, probas):
    preds = probas[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_precision_recall(y_test, probas):
    y_proba = probas[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    auc_precision_recall = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precisión-Recall')
    plt.legend(loc='best')
    plt.show()
    return precision, recall, thresholds