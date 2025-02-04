# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest

# Update and install necessary packages
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install matplotlib
RUN pip3 install SciPy
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install tensorflow
RUN pip3 install scikit-learn
RUN pip3 install statsmodels

COPY ./../../code/ANN/ANN.py /codeBase/
COPY ./../../code/OLS-model/OLS-model.py /codeBase/
COPY ./../../code/ANN/create_ANN.py /codeBase/
COPY ./../../code/OLS-model/create-OLS-model.py /codeBase/
