"""
PricePredictor.py

A machine learning inference module that loads pre-trained regression models
(Linear and Lasso) and predicts the price of a house or car based on input features.

This script provides functionality to:
    - Load serialized models from file paths
    - Preprocess raw input features to numerical format
    - Predict price using the selected model

Author: Nhan Pham  
Email: ptnhanit230104@gmail.com  
Created: 2025-07-26  
Version: 1.0.0
"""

import pickle

class PricePredictor:
    """
    A predictor class for estimating price using Linear or Lasso regression models.

    This class handles loading of models from disk, preprocessing categorical features,
    and generating predictions based on user inputs.
    """

    def __init__(self, linear_model_path, lasso_model_path):
        """
        Initialize the PricePredictor with paths to the serialized models.

        Args:
            linear_model_path (str): Path to the pickled Linear Regression model file.
            lasso_model_path (str): Path to the pickled Lasso Regression model file.
        """
        self.linear_model = self._load_model(linear_model_path)
        self.lasso_model = self._load_model(lasso_model_path)

    def _load_model(self, path):
        """
        Load a machine learning model from a pickle file.

        Args:
            path (str): Path to the pickle file.

        Returns:
            sklearn.base.BaseEstimator: The deserialized machine learning model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def preprocess_features(self, features):
        """
        Convert categorical input features to numeric format using predefined mappings.

        Args:
            features (list): A list of raw input features in the following order:
                [year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]

        Returns:
            list: A list of preprocessed features with categorical values converted to integers.
        """
        fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
        seller_map = {"Individual": 0, "Dealer": 1}
        transmission_map = {"Manual": 0, "Automatic": 1}

        features[3] = fuel_map[features[3]]
        features[4] = seller_map[features[4]]
        features[5] = transmission_map[features[5]]

        return features

    def predict(self, model_type, features):
        """
        Predict the price using the specified regression model.

        Args:
            model_type (str): Type of model to use ('linear' or 'lasso').
            features (list): A list of raw input features.

        Returns:
            float: The predicted price.
        """
        features = self.preprocess_features(features)
        model = self.linear_model if model_type == 'linear' else self.lasso_model
        return model.predict([features])[0]
