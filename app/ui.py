"""
PricePredictionUI.py

A Streamlit-based user interface for predicting house or car prices using 
machine learning models such as Linear and Lasso regression.

This script defines the PricePredictionUI class, which manages:
    - Displaying a title
    - Collecting user input features
    - Triggering prediction callbacks
    - Displaying predicted results

Author: Nhan Pham
Email: ptnhanit230104@gmail.com
Created: 2025-07-26
Version: 1.0.0
"""

import streamlit as st

class PricePredictionUI:
    """
    A Streamlit-based UI class for a house (car) price prediction application.

    This class handles the user interface components such as rendering the title, 
    collecting user inputs for features, showing model selection buttons, and 
    displaying the predicted result.
    """

    def __init__(self):
        """
        Initialize the PricePredictionUI instance.

        Attributes:
            features (list): A list to store user-provided input features for prediction.
        """
        self.features = []

    def render_title(self):
        """
        Render the main title of the application centered at the top of the web interface.
        """
        st.markdown("<h1 align='center'>House Price Prediction App</h1>", unsafe_allow_html=True)

    def get_user_inputs(self):
        """
        Display input fields for the user to enter feature values used in prediction.

        Returns:
            list: A list of input values in the following order:
                [year, price, kms_driven, fuel_type, seller_type, transmission, owner]
                
        Input fields include:
            - Year of manufacture (int)
            - Present price of the car/house (int)
            - Kilometers driven (int)
            - Type of fuel used (str: Petrol, Diesel, CNG)
            - Type of seller (str: Individual, Dealer)
            - Type of transmission (str: Manual, Automatic)
            - Number of owners (int)
        """
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Year", 2000, 2023, 2020)
            kms = st.number_input("Kms Driven", 0, 1_000_000, 10000)
            seller = st.selectbox("Seller Type", ["Individual", "Dealer"])
            owner = st.number_input("Owner", 1, 5, 1)

        with col2:
            price = st.number_input("Present Price", 1000, 1_000_000, 50000)
            fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

        self.features = [year, price, kms, fuel, seller, transmission, owner]
        return self.features

    def show_prediction_buttons(self, callback_linear, callback_lasso):
        """
        Display two buttons allowing the user to choose between Linear and Lasso regression models.

        Args:
            callback_linear (callable): Function to be executed when the Linear model button is clicked.
            callback_lasso (callable): Function to be executed when the Lasso model button is clicked.
        """
        col1, col2 = st.columns(2)
        with col1:
            if st.button("With Linear model", use_container_width=True):
                callback_linear()
        with col2:
            if st.button("With Lasso model", use_container_width=True):
                callback_lasso()

    def display_result(self, price):
        """
        Display the predicted price result to the user in a success message box.

        Args:
            price (float or None): The predicted price to be shown. If None, nothing is displayed.
        """
        if price is not None:
            st.success(f"Predicted Price: ${price:,.2f}")
