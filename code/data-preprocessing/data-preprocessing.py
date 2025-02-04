import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
#import plotly.express as px
from warnings import filterwarnings
filterwarnings('ignore')

data = pd.read_csv('./data/dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
#print(data.head())
print(data.shape)
#print(data.info())
#print(data.isnull().sum())
data.drop_duplicates(inplace=True)
#print(data.duplicated().sum())
print(data.shape)
data = data.rename(columns={'Diabetes_binary': 'Diabetes'})
data=data.reindex(columns=[ 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income','Diabetes'])
# Create the output directory if it doesn't exist 
output_dir = './data/dataset' 
output_dir_pic = 'documentation/pictures'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title("Boxplot to Identify Outliers")
output_file = os.path.join(output_dir_pic, 'Boxplot to Identify Outlierse.png')
plt.savefig(output_file)
plt.show()

# Drop specified columns
data1 = data.drop(columns=['CholCheck', 'Fruits', 'Veggies', 'NoDocbcCost', 'MentHlth', 'CholCheck', 'Smoker'])

# List of skewed features to transform
skewed_features = ['BMI', 'PhysHlth']

# Apply Log Transformation
for feature in skewed_features:
    data[feature] = np.log1p(data[feature])

#Save the DataFrame as a CSV file 
output_file = os.path.join(output_dir, 'joint_data_collection.csv') 
data.to_csv(output_file, index=False)

data_male = data[(data['Sex'] == 1)&(data['Diabetes'] == 1)]
data_male.count()

# frequency of diabetics of all ages for male
data_male = data[data['Sex'] == 1]
male_diabetic_by_age = data_male.groupby('Age')['Diabetes'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(male_diabetic_by_age['Age'], male_diabetic_by_age['Diabetes'] * 100)
plt.xlabel('Age')
plt.ylabel('Percentage of Diabetic Males')
plt.title('Percentage of Diabetic Males by Age')
plt.xticks(male_diabetic_by_age['Age'])
plt.grid(True)

# Create the output directory if it doesn't exist
os.makedirs(output_dir_pic, exist_ok=True)

# Save the plot as an image file
output_file = os.path.join(output_dir_pic, 'diabetic_males_by_age.png')
plt.savefig(output_file)

plt.show()

max_percent_age = male_diabetic_by_age.loc[male_diabetic_by_age['Diabetes'].idxmax()]
print("Maximum percentage of diabetic males is {:.2f}% at age {}.".format(max_percent_age['Diabetes'] * 100, int(max_percent_age['Age'])))

# frequency of diabetics of all ages for female

data_female = data[data['Sex'] == 0]
female_diabetic_by_age = data_female.groupby('Age')['Diabetes'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(female_diabetic_by_age['Age'], female_diabetic_by_age['Diabetes'] * 100)
plt.xlabel('Age')
plt.ylabel('Percentage of Diabetic Females')
plt.title('Percentage of Diabetic Females by Age')
plt.xticks(female_diabetic_by_age['Age'])
plt.grid(True)
# Create the output directory if it doesn't exist
os.makedirs(output_dir_pic, exist_ok=True)

# Save the plot as an image file
output_file = os.path.join(output_dir_pic, 'diabetic_females_by_age.png')
plt.savefig(output_file)

plt.show()

max_percent_age = female_diabetic_by_age.loc[female_diabetic_by_age['Diabetes'].idxmax()]
print("Maximum percentage of diabetic females is {:.2f}% at age {}.".format(max_percent_age['Diabetes'] * 100, int(max_percent_age['Age'])))
