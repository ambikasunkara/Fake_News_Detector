# src/explain.py
import numpy as np
import pandas as pd

def top_influential_words(pipeline, text, top_n=10):
    """
    pipeline is the sklearn Pipeline you saved (tfidf + clf).
    Returns top contributing words with weights.
    """
    # assume pipeline.steps = [("tfidf", vect), ("clf", clf)]
    vect = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']

    clean_x = text  # assume already cleaned by same preprocessing as training
    X_vec = vect.transform([clean_x])

    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        # feature names
        features = np.array(vect.get_feature_names_out())
        # contribution = coef * tfidf value
        contrib = coefs * X_vec.toarray()[0]
        top_idx = np.argsort(contrib)[-top_n:][::-1]
        return list(zip(features[top_idx], contrib[top_idx]))
    else:
        return []
