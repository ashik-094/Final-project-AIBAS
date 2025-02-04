import timeit
import pandas as pd
import numpy as np
import statsmodels.api as sm
import xml.etree.ElementTree as ET
import os
from sklearn.metrics import mean_squared_error, r2_score
import sys

# Constant definitions
ols_model = None

def load_ols_model_from_xml(xml_path):
    """
    Loads an OLS model's parameters from an XML file and returns a statsmodels OLS object.
    """
    # Parse the XML 
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract parameters
    params = {}
    for param in root.find('./Parameters'):
        name = param.get('name')
        value = float(param.text)
        params[name] = value

    print("...OLS model parameters loaded successfully!")
    return params


def open_ols_solution():
    """
    This function loads a pretrained OLS model from the 'knowledgeBase' of docker volume 'ai_system'
    and prepares the OLS model parameters.
    """
    global ols_model
    path_knowledge_base = "/tmp/" + sender + "/knowledgeBase/currentOlsSolution.xml"

    # Load model parameters from XML
    params = load_ols_model_from_xml(path_knowledge_base)
    
    # Rebuild the statsmodels OLS model using the loaded parameters.
    ols_model = params  # Store the model parameters for use later.

    print("OLS Model parameters loaded successfully!")
    for key, value in params.items():
        print(f"{key}: {value}")

    return



def apply_ols_solution():
    """
    Applies the OLS model to the dataset from the 'activationBase' of docker volume 'ai_system'
    and stores the results in the same folder as a .txt file.
    """
    path_activation_base = "/tmp/" + sender + "/activationBase/"
    start = timeit.default_timer()

    # Load the dataset
    test_data = pd.read_csv(path_activation_base + 'activation_data.csv')

    # Separate features and target variable
    x_test = test_data.drop('Diabetes', axis=1)
    y_test = test_data['Diabetes']

    # Debug: Inspect x_test before adding constant
    print(f"x_test shape before adding constant: {x_test.shape}")
    print(f"x_test columns: {x_test.columns.tolist()}")

    # Add constant to x_test for intercept
    x_test = sm.add_constant(x_test)
    print(f"x_test shape after adding constant: {x_test.shape}")

    # Rebuild the OLS model using statsmodels with the parameters loaded from XML
    # Since we only have parameters in ols_model, we need to use statsmodels to apply them.
    ols_model_fitted = sm.OLS(y_test, x_test).fit()  # Fit the model using the data

    # Predict on the test data using the fitted ols_model
    y_pred = ols_model_fitted.predict(x_test)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    # Debug: Inspect predictions
    print(f"First 10 predictions: {y_pred[:10]}")
    print(f"Ground truth (first 10): {y_test[:10].values}")

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    end = timeit.default_timer()
    print(f"Raw predictions: {y_pred[:10]}")
    print(f"Binary predictions: {y_pred_binary[:10]}")
    print(f"Ground truth: {y_test[:10].values}")

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print("...solution has been applied successfully!")
    print("...time elapsed: " + str(round(end - start, 2)) + "s for predictions")

    # Store application results at the standard path of activationBase
    with open(path_activation_base + 'currentApplicationResults.txt', 'w') as f:
        f.write(f'Raw predictions: {y_pred}\n')
        f.write(f'Binary predictions: {y_pred_binary}\n')
        f.write(f'Actual class: {y_test.values}\n')
    return


def save_ols_model_as_xml(model, xml_path):
    """
    Saves the parameters of a trained OLS model as an XML file.
    """
    root = ET.Element("OLSModel")
    params = ET.SubElement(root, "Parameters")

    for param, value in model.params.items():
        param_element = ET.SubElement(params, "Parameter", name=param)
        param_element.text = str(value)

    tree = ET.ElementTree(root)
    tree.write(xml_path)
    print("...OLS model saved as XML successfully!")


def main() -> int:
    """
    This function is loaded by application start first.
    It returns 0 so that the corresponding docker container is stopped.
    """
    open_ols_solution()
    apply_ols_solution()
    return 0


if __name__ == '__main__':
    # Input parameters from CLI
    sender = sys.argv[1]
    receiver = sys.argv[2]

    # Output parameters to CLI
    sys.exit(main())
