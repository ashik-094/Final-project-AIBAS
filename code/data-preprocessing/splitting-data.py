import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('./data/dataset/joint_data_collection.csv')

# Split the data into features (x) and target (y)
x = data.drop('Diabetes', axis=1)
y = data['Diabetes']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Combine x_train and y_train into a single DataFrame for training data
training_data = pd.concat([x_train, y_train], axis=1)

# Combine x_test and y_test into a single DataFrame for testing data
test_data = pd.concat([x_test, y_test], axis=1)

# Create the output directory if it doesn't exist
output_dir = './data/dataset'
os.makedirs(output_dir, exist_ok=True)

# Save the training data to a CSV file
training_data.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)

# Save the test data to a CSV file
test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

# Save a single data entry from the test data to activation_data.csv
activation_data = test_data.sample(n=1, random_state=42)
activation_data.to_csv(os.path.join(output_dir, 'activation_data.csv'), index=False)

print(f"Training data saved to {os.path.join(output_dir, 'training_data.csv')}")
print(f"Test data saved to {os.path.join(output_dir, 'test_data.csv')}")
print(f"Activation data saved to {os.path.join(output_dir, 'activation_data.csv')}")
