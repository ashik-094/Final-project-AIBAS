import timeit
from os import system, name
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xml.etree.ElementTree as ET
import os
import sys

# Constant definitions
model = None

def load_model_from_xml(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Initialize the model
    model = Sequential()

    # Iterate through the layers in the XML file
    for layer in root.findall('./Layers/Layer'):
        layer_name = layer.get('name', 'dense')
        weights = []
        biases = []

        # Extract weights
        weights_element = layer.find('Weights')
        if weights_element is not None:
            weights = [float(w.text) for w in weights_element.findall('Weight')]

        # Extract biases
        biases_element = layer.find('Biases')
        if biases_element is not None:
            biases = [float(b.text) for b in biases_element.findall('Bias')]

        # Convert to numpy arrays
        weights = np.array(weights)
        biases = np.array(biases)

        # Validate dimensions
        if len(biases) == 0:
            raise ValueError(f"No biases found for layer {layer_name}.")
        if len(weights) % len(biases) != 0:
            raise ValueError(
                f"Mismatch in weights ({len(weights)}) and biases ({len(biases)}) "
                f"for layer {layer_name}. Ensure weights = input_dim * output_dim."
            )

        # Calculate dimensions
        output_dim = len(biases)
        input_dim = len(weights) // output_dim

        print(f"Layer: {layer_name}, Input Dim: {input_dim}, Output Dim: {output_dim}")

        # Reshape weights and add layer to the model
        weights = weights.reshape(input_dim, output_dim)
        model.add(Dense(units=output_dim, activation='relu', input_dim=input_dim))
        model.layers[-1].set_weights([weights, biases])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def openAnnSolution():
    """
    This function opens a pretrained ANN from the 'knowledgeBase' of docker volume 'ai_system'.
    """
    global model
    pathKnowledgeBase = "/tmp/" + sender + "/knowledgeBase/currentAiSolution.xml"
    model = load_model_from_xml(pathKnowledgeBase)
    model.summary()
    print("...solution has been loaded successfully!")
    return

def applyAnnSolution():
    """
    This function applies the ANN loaded to the dataset from the 'activationBase' of docker volume 'ai_system' and stores the results in the same folder as a .txt file.
    """
    pathActivationBase = "/tmp/" + sender + "/activationBase/"
    start = timeit.default_timer()

    # Load the dataset
    test_data = pd.read_csv(pathActivationBase + 'activation_data.csv')

    # Separate features and target variable
    x_test = test_data.drop('Diabetes', axis=1)
    y_test = test_data['Diabetes']

    # Standardize the data
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)

    # Predict on the test data
    y_pred = model.predict(x_test)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    end = timeit.default_timer()

    print(f"Raw predictions: {y_pred[:10]}")
    print(f"Binary predictions: {y_pred_binary[:10]}")
    print(f"Ground truth: {y_test[:10].values}")

    # Store application results at standard path of activationBase
    with open(pathActivationBase + 'currentApplicationResults.txt', 'w') as f:
        f.write(f'Raw predictions: {y_pred}\n')
        f.write(f'Binary predictions: {y_pred_binary}\n')
        f.write(f'Actual class: {y_test.values}\n')
    return

def main() -> int:
    """
    This function is loaded by application start first.
    It returns 0 so that corresponding docker container is stopped.
    For instance, limited raspberry resources can be unleashed and energy can be saved.
    """
    openAnnSolution()
    applyAnnSolution()
    return 0

if __name__ == '__main__':
    # Input parameters from CLI
    sender = sys.argv[1]
    receiver = sys.argv[2]
    
    # Output parameters to CLI
    sys.exit(main())
