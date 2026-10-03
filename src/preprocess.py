"""Track 2 cleaning/preprocessing — importable mirror of notebooks/submission.ipynb §3–4.

The NOTEBOOK is the source of truth (the submission must inline everything);
this module exists so teammates can `from src.preprocess import clean, make_features`
in scratch notebooks and stay consistent with the main pipeline. If you change
anything here, change the same code in submission.ipynb AND prediction.ipynb.

Policy (full reasoning in the notebook §3 and docs/eda_insights.md):
- Impute every blank with a hard-coded TRAIN median/mode, no missing-indicator
  flags: train blanks encode an artificial "blank => never fraud" pattern
  (0/113 merchant_category, 0/124 new_device) and the test set has ~3.5x more
  blanks, so NaN-aware trees would wave through exactly the risky rows.
- Category vocabularies fixed from train; unseen category -> NaN (LightGBM
  handles it natively, never a crash).
- `id` dropped: fraud rate is flat across ID ranges.
"""

import pandas as pd

MERCHANTS = ["cash_transfer", "electronics", "entertainment", "food_delivery", "gaming",
             "grocery", "luxury", "retail", "transport", "travel", "utilities"]
COUNTRIES = ["AU", "GB", "ID", "JP", "MY", "PH", "SG", "TH", "US", "VN"]
CHANNELS  = ["bank_transfer", "card_present", "ecommerce", "mobile_app"]
CAT_VOCAB = {"merchant_category": MERCHANTS, "country": COUNTRIES, "transaction_channel": CHANNELS}
IMPUTE = {"transactions_last_24h": 4.0, "spend_last_24h": 217.515, "account_age": 790.0,
          "new_device": 0.0, "transactions_last_1h": 1.0,
          "merchant_category": "grocery", "country": "SG", "transaction_channel": "card_present"}
FEATURES = ["transaction_amount", "transaction_hour", "merchant_category", "country",
            "transaction_channel", "transactions_last_24h", "spend_last_24h",
            "account_age", "new_device", "transactions_last_1h"]
TARGET, ID_COL = "fraud", "id"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c, v in IMPUTE.items():                             # blanks -> typical value (train constants)
        out[c] = out[c].fillna(v)
    for c, vocab in CAT_VOCAB.items():
        out[c] = pd.Categorical(out[c], categories=vocab)   # unseen category -> NaN, never a crash
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must work on the raw test csv alone, with no other inputs."""
    out = clean(df)
    return out[FEATURES]
