"""
model_trainer.py

This module defines the ModelTrainer class, which sets up and executes a
training pipeline for SMS spam classification. It supports:

- Preprocessing SMS text using a custom TextCleaner
- Vectorization using CountVectorizer or TfidfVectorizer
- Training a Multinomial Naive Bayes classifier
- Hyperparameter search over vectorizer and model parameters
- Cross-validation and metric calculation
- Logging experiments, parameters, metrics, and artifacts to MLflow
- Selecting and retraining the best model based on evaluation metrics
"""

import pandas as pd
import os
import tempfile
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas

import config 
from src.data_prep import inputSmsData
from src.text_cleaning import TextCleaner
from src.utils import update_model_config


mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
class ModelTrainer:
    """
    ModelTrainer sets up and trains a spam SMS classification pipeline.

    Attributes:
        param_grid (dict): Dictionary of hyperparameters to search over.
        skf (StratifiedKFold): Cross-validation splitter.
        client (MlflowClient): MLflow client for experiment tracking.
        data (InputSmsData): Data object containing train/val/test splits.
        pipe (Pipeline): Scikit-learn pipeline for cleaning, vectorizing, and classifying.
        run_ids (list): List of MLflow run IDs for all experiments.
        experiment_name (str): Name of the MLflow experiment.
        best_run_id (str): Run ID of the best-performing experiment.
    """

    def __init__(self, csv=config.RAW_DATA_PATH, param_grid=config.param_grid):
        """
        Initialize the ModelTrainer.

        Args:
            csv (str): Path to raw CSV data.
            param_grid (dict): Grid of hyperparameters to explore.
        """

        self.param_grid = param_grid
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=90)
        
        self.client = mlflow.tracking.MlflowClient()
        self.data = inputSmsData(csv)
        
        self.pipe = Pipeline([
            ("cleaner", TextCleaner()),
            ("vectorizer", TfidfVectorizer()),
            ("clf", MultinomialNB())
        ])

        self.run_ids = []
        self.experiment_name = None
        self.best_run_id = None

        

    def train_experiments(self, experiment_name=None):
        """
        Perform a manual hyperparameter search over vectorizers, ngram ranges,
        and Naive Bayes alpha values. Cross-validate each combination and log
        parameters, metrics, and datasets to MLflow.

        Args:
            experiment_name (str, optional): Name of the MLflow experiment. If
                                             None, an automatic name is set.
        """

        if experiment_name:
            self.experiment_name = experiment_name
        else:   
            self.experiment_name = f'exp_{datetime.now()}'
            print('automatic experiment name set')

        mlflow.set_experiment(self.experiment_name)
        train_input = from_numpy(self.data.X_train.to_numpy(), name="train_data", targets=self.data.y_train.to_numpy())
        val_input = from_numpy(self.data.X_val.to_numpy(), name="val_data", targets=self.data.y_val.to_numpy())
        test_input = from_numpy(self.data.X_test.to_numpy(), name="test_data", targets=self.data.y_test.to_numpy())

        mlflow.log_input(train_input, context="training")
        mlflow.log_input(val_input, context="validation")
        mlflow.log_input(test_input, context="testing")

        #self.data.X_test.to_csv("test_data.csv", index=False)

        # set manual search 
        for vect_class in self.param_grid['vectorizers']:
            for ngram in self.param_grid['ngram_ranges']:
                for max_df in self.param_grid['max_dfs']:
                    for min_df in self.param_grid['min_dfs']:
                        for alpha in self.param_grid['alphas']:
                            
                            scoring = {
                                "accuracy": "accuracy",
                                "f1_macro": "f1_macro",
                                "precision_macro": "precision_macro",
                                "recall_macro": "recall_macro",
                                "roc_auc": "roc_auc"
                            }
                            
                            # vectorizer
                            vect = vect_class(ngram_range=ngram, max_df=max_df, min_df=min_df)
                            
                            # pipeline
                            self.pipe.set_params(vectorizer=vect, clf__alpha=alpha)
                            
                            # run CV
                            
                            # log params & results
                            mlflow.end_run()
                            with mlflow.start_run(
                                run_name = f'params ngram={ngram}, maxdf={max_df}, mindf {min_df}, alpha={alpha}',
                                ):
                                scores = cross_validate(
                                    self.pipe, 
                                    self.data.X_train, 
                                    self.data.y_train, 
                                    cv=self.skf, 
                                    scoring= ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"],
                                    return_train_score=True,
                                )
                                print(scores)
                                            

                                # log params
                                mlflow.log_param("vectorizer", vect_class.__name__)
                                mlflow.log_param("ngram_range", ngram)
                                mlflow.log_param("max_df", max_df)
                                mlflow.log_param("min_df", min_df)
                                mlflow.log_param("alpha", alpha)

                                # log metrics
                                train_metrics = {
                                    "train_accuracy": np.mean(scores["train_accuracy"]),
                                    "train_f1_macro": np.mean(scores["train_f1_macro"]),
                                    "train_precision_macro": np.mean(scores["train_precision_macro"]),
                                    "train_recall_macro": np.mean(scores["train_recall_macro"]),
                                    "train_roc_auc": np.mean(scores["train_roc_auc"]),
                                }

                                test_metrics = {
                                    "test_accuracy": np.mean(scores["test_accuracy"]),
                                    "test_f1_macro": np.mean(scores["test_f1_macro"]),
                                    "test_precision_macro": np.mean(scores["test_precision_macro"]),
                                    "test_recall_macro": np.mean(scores["test_recall_macro"]),
                                    "test_roc_auc": np.mean(scores["test_roc_auc"]),
                                }

                                # log metrics
                                for key, value in train_metrics.items():
                                    mlflow.log_metric(key, value)

                                for key, value in test_metrics.items():
                                    mlflow.log_metric(key, value)

                                run_id = mlflow.active_run().info.run_id
                                self.run_ids.append(run_id)
                                
                                print(f"{vect_class.__name__}, ngram={ngram}, max_df={max_df}, min_df={min_df}, alpha={alpha}, recall={np.mean(scores["test_recall_macro"]):.3f}")
        mlflow.end_run()

    def select_best_model(self):
        """
        Select the best-performing run based on test precision and recall.

        Returns:
            str: Run ID of the best experiment.
        """
        if not self.run_ids: 
            print('no runs found in experiment. Did you run "train_experiments()"?')
            return
        
        best_run = mlflow.search_runs(experiment_names=[self.experiment_name],order_by=["metrics.test_precision_macro DESC", "metrics.test_recall_macro DESC"])
        self.best_run_id = best_run['run_id'][0]
        return self.best_run_id
        

        return self.best_run_id

    def train_best_model(self):
        """
        Retrain the best model from selected hyperparameters on the full training
        set, evaluate on validation set, log metrics, confusion matrix, dataset,
        and the trained model to MLflow.
        
        Returns:
            Pipeline: Trained scikit-learn pipeline.
        """

        self.best_run_id = self.select_best_model()
        best_run = self.client.get_run(self.best_run_id)
        best_params = best_run.data.params

        vectorizer_map = {
            "CountVectorizer": CountVectorizer,
            "TfidfVectorizer": TfidfVectorizer
                }
        
        vect_class = vectorizer_map[best_params["vectorizer"]]
        vect = vect_class(
            ngram_range=eval(best_params["ngram_range"]),
            max_df=float(best_params["max_df"]),
            min_df=int(best_params["min_df"])
            )
    
        # update pipeline
        self.pipe.set_params(vectorizer=vect, clf__alpha=float(best_params["alpha"]))
        self.pipe.fit(self.data.X_train, self.data.y_train)

        # fit model
        with mlflow.start_run(run_name="best_model") as run:
            self.pipe.fit(self.data.X_train, self.data.y_train)
            y_val_pred = self.pipe.predict(self.data.X_val)
            y_val_prob = self.pipe.predict_proba(self.data.X_val)[:, 1]

            precision = precision_score(self.data.y_val, y_val_pred, average="macro")
            f1 = f1_score(self.data.y_val, y_val_pred, average="macro")
            recall = recall_score(self.data.y_val, y_val_pred, average="macro")
            roc_auc = roc_auc_score(self.data.y_val, y_val_prob)

            # Log metrics
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # log cm and test dataset for later predicting
            with tempfile.TemporaryDirectory() as tmpdir:
                cm = confusion_matrix(self.data.y_val, y_val_pred)
                fig, ax = plt.subplots(figsize=(5, 5))
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=["Ham (0)", "Spam (1)"]
                )
                disp.plot(cmap='Blues', ax=ax, colorbar=False)
                ax.set_title("Ham/Spam prediction scores", fontsize=12, pad=15)
                ax.set_xlabel("Predicted Label", fontsize=10)
                ax.set_ylabel("True Label", fontsize=10)
                ax.tick_params(axis='x', labelrotation=0)
                plt.tight_layout()
                cm_path = f"{tmpdir}/cm.png"
                fig.savefig(cm_path, bbox_inches='tight')
                plt.close(fig)

                
                x_path = f"{tmpdir}/test_data_x.csv"
                y_path = f"{tmpdir}/test_data_y.csv"
                pd.DataFrame(self.data.X_test).to_csv(x_path, index=False)
                pd.DataFrame(self.data.y_test).to_csv(y_path, index=False)

                mlflow.log_artifact(cm_path, artifact_path="plots")
                mlflow.log_artifact(x_path, artifact_path="dataset")
                mlflow.log_artifact(y_path, artifact_path="dataset")
            
            # log model
            registered_model_name = f"{list(best_params.values())}_{datetime.now()}"
            mlflow.sklearn.log_model(
                self.pipe, 
                name="model", 
                input_example=self.data.X_val[:5].to_frame(name="text"), 
                registered_model_name=f'{registered_model_name}')
            update_model_config(name = registered_model_name, version=1)
            print(f"Final model trained and logged. Precision: {precision:.3f}, recall: {recall:.3f}, ROC-AUC: {roc_auc}")
            return self.pipe
    


