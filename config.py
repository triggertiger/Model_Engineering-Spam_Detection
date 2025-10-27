from pathlib import Path
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# file system
current_dir = Path(os.getcwd()).resolve()
RAW_DATA_PATH = current_dir / "data" / "SMSSpamCollection.csv"

# mlflow tracking
MLFLOW_TRACKING_URI = os.path.join(current_dir, "mlruns")

#
SNS_STYLE = "whitegrid"
COLOR_MAP = "viridis" 

param_grid = {
    "vectorizers" : [CountVectorizer, TfidfVectorizer],
    "ngram_ranges" : [(1,1), (1,2)],                      # vectorizer relates combinations of 1 word or 1&2 words
    "max_dfs" : [0.95, 0.9],                              #  ignore too often words
    "min_dfs" : [1, 5],                                   # ignore too rare words
    "alphas" : [1.0, 0.1],                                # NB alpha: smoothing factor for bias handling 
    "scores": ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
}
