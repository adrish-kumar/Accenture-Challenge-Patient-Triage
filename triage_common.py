"""
triage_common.py
=================
Shared helper(s) used by both the training and prediction scripts.

IMPORTANT: this function must live in its own importable module (not inside
train_triage_model.py's __main__ block) so that joblib can correctly
pickle/unpickle the ColumnTransformer that references it, regardless of
which script loads the model later.
"""

TEXT_FEATURE = "Chief_complain"

def select_text(df_):
    """Extract the free-text chief-complaint column for the TF-IDF step."""
    return df_[TEXT_FEATURE]
