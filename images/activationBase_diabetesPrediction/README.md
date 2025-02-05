# Dealing with this local image

The current activation refers to image called `activation_data.csv`.

## Build local docker image manually with `Dockerfile` for Over-The-Air-Deployment of relevant data.

### Build local docker image manually with `Dockerfile`.

1. Build docker image from Dockerfile specified.

    ```
    docker build --tag ashikzaman43/activationbase_diabetesprediction:latest .
    ```

1. Have a look on the image created.    
    
    ```
    docker run -it --rm ashikzaman43/activationbase_diabetesprediction:latest sh
    ```

### Alternatively, build local docker image manually with `yml` file.

1. If not available, yet, create independent volume for being bound to image.

    ```
    docker volume create ai_system
    ```
    
1. Build image with `Docker-compose`.
    
    ```
    Docker-compose build
    ```

### Test local docker image.

1. Start image with `Docker-compose`.
    
    ```
    Docker-compose up
    ```

1. Test your image, e.g. by executing a shell.

    ```
    docker exec -it ashikzaman43/activationbase_diabetesprediction:latest sh
    ```
    
1. Shut down image with `Docker-compose`.
    
    ```
    Docker-compose down
    ```

### Deploy local docker image to dockerhub.
 
1. Push image to `https://hub.docker.com/` of account called `ashikzaman43`.
    
    ```
    docker push ashikzaman43/activationbase_diabetesprediction:latest
    ```
    
# Credits

Dataset is coming from the following repositories:
Diabetes Health Indicators Dataset from kaggle repository [Alex Teboul] (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) licence under CC0: Public Domain. This is created as part of the course ‘M. Grum: Advanced AI-based ApplicaGon Systems’ by the Junior Chair for Business InformaGon Science, esp. AI-based ApplicaGon Systems at University of Potsdam. Our model is a feed-forward Artificial neural network(ANN) with 3 layers including input and output layer. Key performance indicator for this model was precision, recall and f1-score. The diabetes_binary_50550_split_health_indicatorts_BRFSS2015.csv was considered to train and test upon as this dataseet was clean and have 70,692 survey responses. The commitment to the ‘AGPL-3.0 license’ was kept as it is, it was forked from (https://github.com/MarcusGrum/AI-CPS). 