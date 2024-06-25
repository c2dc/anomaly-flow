from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score 


def eval_learning(y_test, preds):
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    missrate = fn / (fn + tp)
    fallout = fp / (fp + tn)
    auc = roc_auc_score(y_test, preds)
    f2_value = fbeta_score(y_test, preds, beta=2)

    return acc, rec, prec, f1, mcc, missrate, fallout, auc, f2_value
