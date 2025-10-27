"""
predict_pipeline.py

This module provides classes for loading the latest registered spam 
classification model from MLflow and predicting the class of new SMS messages.

Includes:
- Message: a dataclass for storing individual message predictions
- Predictor: loads the model and applies it to new messages
"""

from dataclasses import dataclass
import pandas as pd
import tempfile
import os

import mlflow

import config 
from src.data_prep import inputSmsData
from src.text_cleaning import TextCleaner
from src.utils import read_model_config

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


@dataclass
class Message:
    """
    Holds a single message and its classification outcome.
    """
    text: str
    prob: float
    tag: str = 'low_risk' 
    class_: str = 'clear'


class Predictor:
    """
    Loads the last registered spam classifier model from MLflow and predicts
    classifications for new SMS messages.

    Attributes:
        config (dict): Model configuration read from YAML file.
        client (MlflowClient): MLflow client for fetching model info and artifacts.
        model (Pipeline): Loaded scikit-learn pipeline for prediction.
        run_id (str): MLflow run ID associated with the registered model.
        experiment_id (str): MLflow experiment ID.
        test_data (pd.DataFrame): Previously logged test dataset.
    """

    def __init__(self):
        self.config = read_model_config()
        self.client = mlflow.MlflowClient()

        self.model, self.run_id, self.experiment_id = self.load_model()
        self.test_data = self.load_logged_test_data()

    def load_model(self):
        model_name = self.config['name']
        version = self.config['last_registered_version']
        model_uri = f"models:/{model_name}/{version}"

        model_version = self.client.get_model_version(name=model_name, version=version)
        run_id = model_version.run_id
        model = mlflow.sklearn.load_model(model_uri)
        run = self.client.get_run(run_id)
        experiment_id = run.info.experiment_id

        return model, run_id, experiment_id

    def predict_messages(self, messages: list[str], threshold: str = "medium"):
        """
        Predict the class of a list of messages and assign risk tags.

        Args:
            messages (list[str]): List of SMS messages to classify.
            threshold (str): Risk threshold level ("low", "medium", "high").
                             Determines probability cutoff for 'spam' classification.

        Returns:
            list[Message]: List of Message objects containing predictions and tags.
        """
          
        thres_probability_map = {"low": 0.2, "medium":0.3, 'high': 0.4 }
        chosen_risk = thres_probability_map[threshold]

        df = pd.Series(messages)
        probabilities = self.model.predict_proba(df)[:,1]
        results = []

        for text, prob in zip(messages, probabilities):
            # set the class based on the risk threshold chosen
            class_ = "spam" if prob > chosen_risk else "ham"

            # set tags for the 'ham' messages already classified
            if class_ == 'ham':
                if prob < 0.25 * chosen_risk:
                    tag = 'high certainty'
                elif prob < 0.5 * chosen_risk:
                    tag = 'medium certainty'
                else: 
                    tag = 'high risk'
            else: 
                tag = 'high certainty'
            
            results.append(Message(text=text, prob=prob, class_=class_, tag=tag))
        
        return results
    
    def load_logged_test_data(self):
        """
        Load the test dataset previously logged to MLflow for reference.

        Returns:
            pd.DataFrame: Test dataset as a pandas DataFrame.
        """
        tmpdir = tempfile.mkdtemp()
        path = self.client.download_artifacts(self.run_id, "dataset/test_data.csv", tmpdir)
        test_data = pd.read_csv(path)
        return test_data
        

    