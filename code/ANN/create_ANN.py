import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(training_data_path, test_data_path):
    """Loads training and test datasets and returns standardized features and labels."""
    training_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)
    
    x_train = training_data.drop('Diabetes', axis=1)
    y_train = training_data['Diabetes']
    
    x_test = test_data.drop('Diabetes', axis=1)
    y_test = test_data['Diabetes']
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, y_train, x_test, y_test

def build_model(input_shape):
    """Creates and compiles a Sequential ANN model."""
    model = Sequential([
        Dense(32, input_dim=input_shape, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
    """Trains the model and returns the training history."""
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

def evaluate_model(model, x_test, y_test):
    """Evaluates the trained model and returns performance metrics."""
    y_pred = model.predict(x_test)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'f1_score': f1_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary)
    }

def save_model_as_xml(model, filepath):
    """Saves the model parameters as an XML file."""
    root = ET.Element("TensorFlowModel")
    layers = ET.SubElement(root, "Layers")
    
    for layer in model.layers:
        layer_element = ET.SubElement(layers, "Layer", name=layer.name)
        weights, biases = layer.get_weights()
        
        weights_element = ET.SubElement(layer_element, "Weights")
        biases_element = ET.SubElement(layer_element, "Biases")
        
        for weight in weights.flatten():
            weight_element = ET.SubElement(weights_element, "Weight")
            weight_element.text = str(weight)
        
        for bias in biases.flatten():
            bias_element = ET.SubElement(biases_element, "Bias")
            bias_element.text = str(bias)
    
    tree = ET.ElementTree(root)
    tree.write(filepath)

def save_performance_metrics(metrics, filepath):
    """Saves the model evaluation metrics to a text file."""
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(key.capitalize() + ': ' + str(value) + '\n')

def plot_training_performance(history, output_dir):
    """Plots and saves training/validation accuracy and loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir + '/training_validation_accuracy.png')
    plt.savefig(output_dir + '/training_validation_accuracy.pdf', format='pdf')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir + '/training_validation_loss.png')
    plt.savefig(output_dir + '/training_validation_loss.pdf', format='pdf')
    plt.show()

def main():
    """Main function to execute the training and evaluation pipeline."""
    output_dir = "/tmp/" + sender + "/learningBase"
    output_dir_model = "/tmp/" + sender + "/knowledgeBase"
    training_data_path = "/tmp/" + sender + "/learningBase/train/training_data.csv"
    test_data_path = "/tmp/" + sender + "/learningBase/validation/test_data.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)
    
    x_train, y_train, x_test, y_test = load_data(training_data_path, test_data_path)
    model = build_model(x_train.shape[1])
    history = train_model(model, x_train, y_train, x_test, y_test)
    metrics = evaluate_model(model, x_test, y_test)
    
    print(metrics)
    
    save_model_as_xml(model, output_dir_model + '/currentAiSolution.xml')
    save_performance_metrics(metrics, output_dir + '/performance_metrics.txt')
    plot_training_performance(history, output_dir)
    
    return 0

if __name__ == '__main__':
    sender = sys.argv[1]
    receiver = sys.argv[2]
    sys.exit(main())
