"""
main.py

Entry point for the House/Car Price Prediction App using Streamlit.

This script connects the UI layer (`PricePredictionUI`) with the ML inference logic (`PricePredictor`). It:
    - Initializes the UI and models
    - Collects user inputs
    - Handles predictions using Linear or Lasso regression
    - Displays results to the user

Author: Nhan Pham  
Email: ptnhanit230104@gmail.com  
Created: 2025-07-26  
Version: 1.0.0
"""

from predictor import PricePredictor
from ui import PricePredictionUI

def main():
    """
    Main function to run the Streamlit app.

    Initializes UI and model predictor components, collects inputs,
    performs prediction based on user's model choice, and displays results.
    """
    ui = PricePredictionUI()
    predictor = PricePredictor(
        linear_model_path="model/linear_model.pkl",
        lasso_model_path="model/lasso_model.pkl"
    )

    ui.render_title()
    features = ui.get_user_inputs()

    result = {"price": None}

    def handle_linear():
        """Handle prediction using the Linear Regression model."""
        result["price"] = predictor.predict('linear', features)

    def handle_lasso():
        """Handle prediction using the Lasso Regression model."""
        result["price"] = predictor.predict('lasso', features)

    ui.show_prediction_buttons(handle_linear, handle_lasso)
    ui.display_result(result["price"])

if __name__ == "__main__":
    main()
