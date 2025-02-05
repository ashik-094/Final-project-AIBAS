import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

def load_data(training_data_path, test_data_path):
    """Loads training and test datasets and returns features and labels."""
    training_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)
    
    x_train = training_data.drop('Diabetes', axis=1)
    y_train = training_data['Diabetes']
    
    x_test = test_data.drop('Diabetes', axis=1)
    y_test = test_data['Diabetes']
    
    return x_train, y_train, x_test, y_test

def build_model(x_train, y_train):
    """Creates and fits the OLS model."""
    x_train = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_train).fit()
    return model

def evaluate_model(model, x_test, y_test):
    """Evaluates the trained model and returns performance metrics."""
    x_test = sm.add_constant(x_test)
    y_pred = model.predict(x_test)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'f1_score': f1_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary)
    }

def save_model_as_xml(model, filepath):
    """Saves the model parameters as an XML file."""
    root = ET.Element("OLSModel")
    params = ET.SubElement(root, "Parameters")
    
    for param, value in model.params.items():
        param_element = ET.SubElement(params, "Parameter", name=param)
        param_element.text = str(value)
    
    tree = ET.ElementTree(root)
    tree.write(filepath)

def save_performance_metrics(metrics, filepath):
    """Saves the model evaluation metrics to a text file."""
    with open(filepath, 'w') as f:
        for key, value in metrics.items():
            f.write(key.capitalize() + ': ' + str(value) + '\n')

def plot_training_performance(model, x_train, y_train, x_test, y_test, y_pred, output_dir):
    """Plots and saves training/testing curves, residual plot, Q-Q plot, and actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.values, label='Training Data')
    plt.plot(model.predict(sm.add_constant(x_train)), label='Model Prediction on Training Data')
    plt.plot(y_test.values, label='Testing Data')
    plt.plot(y_pred, label='Model Prediction on Testing Data')
    plt.xlabel('Index')
    plt.ylabel('Diabetes')
    plt.title('Training and Testing Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_testing_curves.png'))
    plt.savefig(output_dir + '/training_testing_curves.pdf', format='pdf')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_test - y_pred)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'))
    plt.savefig(output_dir + '/residual_plot.pdf', format='pdf')
    plt.show()
    
    sm.qqplot(model.resid, line='45')
    plt.title('Q-Q Plot')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'qq_plot.png'))
    plt.savefig(output_dir + '/qq_plot.pdf', format='pdf')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.savefig(output_dir + '/actual_vs_predicted.pdf', format='pdf')
    plt.show()

def main():
    """Main function to execute the training and evaluation pipeline."""
    output_dir = "/tmp/" + sender + "/learningBase"
    output_dir_model = "/tmp/" + sender + "/knowledgeBase"
    training_data_path = "/tmp/" + sender + "/learningBase/train/training_data.csv"
    test_data_path = "/tmp/" + sender + "/learningBase/validation/test_data.csv"
    model_summary_path = os.path.join(output_dir, "model_summary.txt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    x_train, y_train, x_test, y_test = load_data(training_data_path, test_data_path)
    model = build_model(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    
    print(metrics)
    
    save_model_as_xml(model, os.path.join(output_dir_model, 'currentOlsSolution.xml'))
    save_performance_metrics(metrics, os.path.join(output_dir, 'performance_metrics.txt'))
    #plot_training_performance(model, x_train, y_train, x_test, y_test, model.predict(sm.add_constant(x_test)), output_dir)

    # Save the model summary to a text file
    with open(model_summary_path, "w") as f:
        f.write(model.summary().as_text())

    # Print the model summary in the console
    print(model.summary())

    return 0

if __name__ == '__main__':
    sender = sys.argv[1]
    receiver = sys.argv[2]
    sys.exit(main())
