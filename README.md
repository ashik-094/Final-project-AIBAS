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
- **HeavyAlcoholConsumption:** Indicates heavy drinking.
- **Diabetes_012:** This is the target variable. It indicates no diabetes (0), prediabetes (1), and diabetes (2). (We work only with diabetes and no diabetes)

### Utility

This dataset is primarily used for statistical analysis and machine learning models to understand and predict diabetes based on various health indicators. It's ideal for tasks such as classification, where the aim is to predict the likelihood of diabetes based on the health indicators.

### This is based on the Docker Installation Guide for Windows.

## 1) Download and Install Docker Desktop
Download the latest Docker Desktop for Windows from the official Docker website.
Run the installer and follow the installation instructions.
Ensure WSL 2 (Windows Subsystem for Linux) is enabled when prompted.
2) Verify Docker Installation
Run the following command in PowerShell or Command Prompt:
```
docker run hello-world
```
This will confirm that Docker is installed and working correctly.

3) Enable Docker to Start on Boot
Open Docker Desktop.
Go to Settings → General.
Check "Start Docker Desktop when you log in".
4) Create a Docker Volume
To create the ai_system volume for persistent storage, run:
```
docker volume create ai_system
```
5) Run Docker Containers Using WSL 2 (Optional)
If you use WSL 2, ensure it’s set as the default backend by running:
```
wsl --set-default-version 2
```
# Execution Workflow

This section outlines the steps involved in running our code, from creating volumes to retrieving output files.

### Steps

1. **Create a Volume**: Start by creating a volume where all output files will be stored. If you already created it above then no need. 
2. **Build Images**: Next, build the Docker images needed for the project. There are a total of 4 docker images. Already these images are builed and pushed in docker hub with public access rights. 

Build the Docker images using specific Dockerfiles:
```
docker build -f images/codeBase_diabetesPrediction/Dockerfile -t ashikzaman43/codebase_diabetesprediction:latest .
```
```
docker build -f images/activationBase_diabetesPrediction/Dockerfile -t ashikzaman43/activationbase_diabetesprediction:latest .
```
```
docker build -f images/knowleddgeBase_diabetesPrediction/Dockerfile -t ashikzaman43/knowledgebase_diabetesprediction:latest .
```
```
docker build -f images/learningBase_diabetesPrediction/Dockerfile -t ashikzaman43/learningbase_diabetesprediction:latest .
```
3. **Push Images to Docker Hub**: Once the images are built, push them to Docker Hub to make them accessible for further use.

Push the Docker images to Docker Hub:
```
docker push ashikzaman43/codebase_diabetesprediction:latest
```

```
docker push ashikzaman43/activationbase_diabetesprediction:latest
```

```
docker push ashikzaman43/knowledgebase_diabetesprediction:latest
```

```
docker push ashikzaman43/learningbase_diabetesprediction:latest
```
4. **Run the YML File**: To start the processing, execute the YML file from the scenario folder.

From the scenario folder, run the Docker Compose file:
To predict single test data from activation dataset using ANN:
```
docker-compose -f docker-compose-ai.yml up
```
To train and evaluate the model again and save the model considering ANN:
```
docker-compose -f docker-compose-ai-create-eval.yml up
```
To predict single test data from activation dataset using OLS:
```
docker-compose -f docker-compose-ols.yml up
```
To train and evaluate the model again and save the model considering OLS:
```
docker-compose -f docker-compose-ols-create-eval.yml up
```

5. **Store the Outcomes**: The process's outputs are stored in the created volume. They can be in various formats, such as `.txt`, `.xml`, `.png`, `.pdf`, or `.csv`.
6. **Transfer Files**: Finally, move the files from the volume to your local machine for further analysis or use.
Move the files from the volume to your local machine:

```
docker cp <container_name_or_id>:<source_path_inside_container> <destination_path_on_host>
```
#### Example
```
docker cp scenarios-code_base_test-1:tmp/test/learningBase "F:/AIBAS/Final project/Final-project-AIBAS/documentation/"
```
in the documentation directory results of ANN, OLS are saved.

### Docker Images in DocerHub

If you want to see the Docker images for this project, please visit my DockerHub repository at the following link:

https://hub.docker.com/repositories/ashikzaman43

By following these steps, you can efficiently manage the workflow and ensure that the data is processed and stored correctly. Data preprocess and splitting were done on local machine. The code is in the directory of code/data-preprocessing. 

# Credits
Dataset is coming from the following repositories:
Diabetes Health Indicators Dataset from kaggle repository [Alex Teboul] (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) licence under CC0: Public Domain. This is created as part of the course ‘M. Grum: Advanced AI-based ApplicaGon Systems’ by the Junior Chair for Business InformaGon Science, esp. AI-based ApplicaGon Systems at University of Potsdam.
