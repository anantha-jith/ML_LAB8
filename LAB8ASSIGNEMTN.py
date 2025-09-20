# ecg_analysis_pipeline.py
# Full end-to-end ECG dataset analysis (load, preprocess, train, evaluate)
# Save this file and run in a Python environment with pandas, scikit-learn, matplotlib installed.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

DATA_PATH = r"C:\Users\Haren A\OneDrive\Desktop\labnndl\ecg.csv.xlsx"
  

def load_data(path=DATA_PATH):
    """Load dataset (tries CSV first, then Excel). Returns dataframe or raises error."""
    if os.path.exists(path):
        try:
            if path.lower().endswith(".csv") or path.lower().endswith(".txt"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")

def quick_inspect(df):
    """Return quick information about dataframe: shape, dtypes, missing, head"""
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "missing_count": df.isna().sum().to_dict(),
        "head": df.head(8)
    }
    return info

def preprocess(df, target_column=None):
    """Basic preprocessing:
       - If target_column is None, tries to infer a target column (common names) or uses last column.
       - Encodes categorical target if needed.
       - Drops fully empty columns, fills or drops missing values (simple strategy).
       - Returns X, y, scaler (fitted), classes, and the name of target column.
    """
    df = df.copy()
    # drop fully empty columns
    df = df.dropna(axis=1, how='all')
    # infer target column
    if target_column is None:
        possible = [c for c in df.columns if c.lower() in ("label","target","class","y","diagnosis","arrhythmia","beat","annotation")]
        if len(possible) > 0:
            target_column = possible[0]
        else:
            target_column = df.columns[-1]
    # separate X and y
    y = df[target_column].copy()
    X = df.drop(columns=[target_column])
    # handle missing: drop rows with missing target, fill numeric missing with median
    non_missing_mask = ~y.isna()
    df = df[non_missing_mask]
    X = X.loc[non_missing_mask, :].copy()
    y = y.loc[non_missing_mask].copy()
    for col in X.columns:
        if X[col].dtype.kind in 'biufc':  # numeric
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("missing")
    # encode categorical features minimally (one-hot for small cardinality)
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name=='category']
    for c in cat_cols:
        if X[c].nunique() > 30:
            X = X.drop(columns=[c])
        else:
            dummies = pd.get_dummies(X[c], prefix=c, drop_first=True)
            X = pd.concat([X.drop(columns=[c]), dummies], axis=1)
    # encode target
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        classes = le.classes_.tolist()
    else:
        y_enc = y.values
        classes = None
    # scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, y_enc, scaler, classes, target_column

def fit_and_eval(X, y, random_state=42):
    """Split, train three classifiers, return trained models and evaluation metrics"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state,
        stratify=y if len(np.unique(y))>1 else None
    )
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "mlp": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=random_state)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        roc_auc = None
        if len(np.unique(y)) == 2:
            try:
                probs = model.predict_proba(X_test)[:,1]
                roc_auc = roc_auc_score(y_test, probs)
            except Exception:
                roc_auc = None
        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": cr,
            "roc_auc": roc_auc
        }
    return results, X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Simple confusion matrix plot using matplotlib (single plot)"""
    plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     )
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, probs, title="ROC Curve"):
    """Single ROC plot"""
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()

# Run pipeline
if _name_ == "_main_":
    df = load_data()
    info = quick_inspect(df)
    print("Data shape:", info["shape"])
    print("Columns:", info["columns"])
    print("Missing counts (per col):", info["missing_count"])
    print("Sample rows:")
    print(info["head"])

    X, y, scaler, classes, target_col = preprocess(df, target_column=None)
    print("Inferred target:", target_col, " (classes: ", classes, ")")
    print("Feature sample:")
    print(X.head())

    results, X_train, X_test, y_train, y_test = fit_and_eval(X, y)

    # Summarize results and save
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "model": name,
            "accuracy": res["accuracy"],
            "roc_auc": res["roc_auc"],
            "test_samples": len(y_test)
        })
    summary_df = pd.DataFrame(summary_rows)
    out_dir = "./ecg_analysis_results"
    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)
    import json
    class_reports = {name: res["classification_report"] for name,res in results.items()}
    with open(os.path.join(out_dir, "classification_reports.json"), "w") as f:
        json.dump(class_reports, f, indent=2)
    print("Saved results to", out_dir)

    # show confusion matrix for best model
    best = summary_df.sort_values("accuracy", ascending=False).iloc[0]["model"]
    print("Best model:", best)
    best_res = results[best]
    plot_confusion_matrix(best_res["confusion_matrix"], title=f"Confusion Matrix: {best}")
    if best_res["roc_auc"] is not None:
        probs = best_res["model"].predict_proba(X_test)[:,1]
        plot_roc_curve(y_test, probs, title=f"ROC Curve: {best}")
    else:
        print("ROC AUC not computed (non-probabilistic model or multi-class).")
