from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


def oof(model, X, y, cv=5, seed=42):
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]


def cv_auc(model, X, y, cv=5):
    a = roc_auc_score(y, oof(model, X, y, cv=cv))
    print(f"OOF AUC: {a:.4f}")
    return a
