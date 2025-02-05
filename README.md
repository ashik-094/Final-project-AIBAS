# Welcome to the Project AI-Powered Diabetes Prediction

## Project Overview

Our project focuses on predicting diabetes using artificial neural networks (ANN). Diabetes is a chronic condition that affects the body's ability to regulate blood glucose, significantly impacting quality of life and life expectancy.

### Data Acquisition and Preprocessing

We utilized a large dataset from Kaggle, accessed via the Kaggle API key, which featured labeled data in three classes. For our purposes, we refined the dataset to focus on two classes—diabetic and non-diabetic. We opted to manually select features to drop rather than using automated feature selection techniques. Due to notable outliers in BMI and physical health status indicators, we applied log transformations to normalize the data, which enhances model accuracy. All data was standardized prior to training.

### Model Architecture

We designed a binary classification model using Keras's Sequential API. Our model consists of a three-layer feedforward neural network, employing ReLU activation functions in the hidden layers and a Sigmoid output layer. The model is optimized using the Adam optimizer and trained with binary cross-entropy loss to effectively predict diabetic and non-diabetic classes.


## Dataset Overview

This dataset is compiled and shared by a user on Kaggle and is based on the Behavioral Risk Factor Surveillance System (BRFSS) data from the Centers for Disease Control and Prevention (CDC). BRFSS is a United States health-related telephone survey that collects state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services.

### Key Features

Some of the key features in this dataset include:

- **Smoking:** Indicates whether the individual currently smokes.
- **PhysicalActivity:** Indicates whether the individual engages in physical activity.
- **Fruits:** Consumption frequency of fruits.
- **Veggies:** Consumption frequency of vegetables.
- **HeavyAlcoholConsumption:** Indicates heavy drinking.
- **NoDocbcCost:** Indicates if there were any times a person needed to see a doctor but could not because of cost.
- **Diabetes_012:** This is the target variable. It indicates no diabetes (0), prediabetes (1), and diabetes (2). (We work only with diabetes and no diabetes)

### Utility

This dataset is primarily used for statistical analysis and machine learning models to understand and predict diabetes based on various health indicators. It's ideal for tasks such as classification, where the aim is to predict the likelihood of diabetes based on the health indicators.


# Execution Workflow

This section outlines the steps involved in running our code, from creating volumes to retrieving output files.

### Steps

1. **Create a Volume**: Start by creating a volume where all output files will be stored.
2. **Build Images**: Next, build the Docker images needed for the project.
3. **Push Images to Docker Hub**: Once the images are built, push them to Docker Hub to make them accessible for further use.
4. **Run the YML File**: To start the processing, execute the YML file from the scenario folder.
5. **Store the Outcomes**: The process's outputs are stored in the created volume. They can be in various formats, such as `.txt`, `.xml`, `.png`, `.pdf`, or `.csv`.
6. **Transfer Files**: Finally, move the files from the volume to your local machine for further analysis or use.

By following these steps, you can efficiently manage the workflow and ensure that the data is processed and stored correctly.

1. Build docker image from Dockerfile specified.

```bash
docker build --tag ashikzaman43/learningbase_diabetesprediction:latest .

The following is the latest:

docker build -f images/codeBase_diabetesPrediction/Dockerfile -t ashikzaman43/codebase_diabetesprediction:latest .

(PowerShell in F:\AIBAS\Project AIBAS)

PS F:\AIBAS\Project AIBAS> docker build -f images/codeBase_diabetesPrediction/Dockerfile -t ashikzaman43/codebase_diabetesprediction:latest .


