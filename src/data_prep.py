"""
data_prep.py

This module handles loading and preparing SMS spam data for modeling. 
It provides functionality to read the raw CSV dataset and split it 
into training, validation, and test sets.
"""

import pandas as pd
import os
import config 

from sklearn.model_selection import train_test_split


class inputSmsData:
    """
    Class for loading and splitting SMS spam data.

    Attributes:
        df (pd.DataFrame): Raw dataframe containing SMS texts and labels.
        X_train (pd.Series): Training set features (SMS texts).
        X_val (pd.Series): Validation set features.
        X_test (pd.Series): Test set features.
        y_train (pd.Series): Training set labels (ham/spam).
        y_val (pd.Series): Validation set labels.
        y_test (pd.Series): Test set labels.
    """

    def __init__(self, csv=config.RAW_DATA_PATH):
        """
        Initialize InputSmsData instance by reading CSV and splitting the data.

        Args:
            csv (str): Path to the raw CSV data file. Defaults to config.RAW_DATA_PATH.
        """
        self.df = pd.read_csv(
            csv, 
            delimiter='\t', 
            header=None, 
            encoding='utf-8',
            names=['label', 'text']
            )
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()

        
    def split_data(self, test_size=0.05):
        """
        Split the data into train, validation, and test sets.

        The splitting is done in two stages:
        1. Hold out a portion of data for validation + test.
        2. Split holdout data into validation and test sets.

        Args:
            test_size (float): Fraction of data to use for test set. Validation
                               size is automatically set to test_size * 4.

        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            self.df["text"], self.df["label"],
            test_size=test_size * 5, stratify=self.df["label"], random_state=60
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout,
            test_size= test_size * 4, stratify=y_holdout, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test





