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


## Step 2: Build Docker Images

Build the Docker images using specific Dockerfiles:

```
docker build -f images/codeBase_diabetesPrediction/Dockerfile -t ashikzaman43/codebase_diabetesprediction:latest .
```
#### Do the above for all the images

## Step 3: Push Docker Images to Docker Hub

Push the Docker images to Docker Hub:
```
docker push ashikzaman43/learningbase_diabetesprediction:latest
```
#### Do this for all the images

## Step 4: Run Docker Compose

From the scenario folder, run the Docker Compose file:

```
docker-compose -f docker-compose-ai.yml up
```

#### Do this for other .yml files

## Step 5: Outcome Storage
The outcome will be stored in the volume as .txt, .xml, .png, .pdf, or .csv files.

### Step 6: Move Files from Volume to Local Machine
Move the files from the volume to your local machine:

```
docker cp <container_name_or_id>:<source_path_inside_container> <destination_path_on_host>
```
#### Example
```
docker cp scenarios-code_base_test-1:tmp/test/learningBase "F:/AIBAS/Final project/Final-project-AIBAS/documentation/"
```



### Docker Images

If you want to see the Docker images for this project, please visit my DockerHub repository at the following link:

(https://hub.docker.com/repositories/ashikzaman43)




