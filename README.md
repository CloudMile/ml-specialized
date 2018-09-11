# ML Specialized Project

CloudMile project for google ml-specialist 

## DNNRegressor/DNNLinearCombinedRegressor on Rossmann data 

Rossmann data is a collection of time-series data for the prediction of sales in the next 6 months for each store. The greatest challenge is whether the model can accurately obtain the mean sales for each store, as well as the periodic and seasonal information.

- From kaggle [Rossmann](https://www.kaggle.com/c/rossmann-store-sales)
- See the [rossman.ipynb](rossmann/rossmann.ipynb) for details

## Custom tf.estimator.Estimator for KKBOX Music Recommendation Engine

KKBOX data is the data for a music recommendation challenge. Through personal records, we attempt to predict the scoring for songs as well as the click-through rate (CTR). 

- For this personalized music recommendation, since the target column is binary, we tackle this task as a classification problem, i.e. this is a binary classification problem.
- From kaggle [WSDM - KKBox's Music Recommendation Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data)
- See the [kkbox.ipynb](kkbox/kkbox.ipynb) for details


## Train Data and Model Check Points on GCP

- Rossmann 
    - data: gs://ml-specialized/rossmann/data
        ```
        │  store.csv
        │  store_states.csv
        │  test.csv
        └─train.csv
        ```
    - model: gs://ml-specialized/rossmann/models
        ```
        │  saved_model.pb
        └─ variables
                variables.data-00000-of-00001
                variables.index
        ```
- KKBOX 
    - data: gs://ml-specialized/kkbox/data
        ```
        │  members.csv
        │  songs.csv
        │  song_extra_info.csv
        │  test.csv
        └─ train.csv
        ```
    - model: gs://ml-specialized/kkbox/model
        ```
        │  saved_model.pb
        └─ variables
                variables.data-00000-of-00001
                variables.index
        ```
        
 The project can be executed with jupyter notebook through the work flow after data is stored under respective directories. You will also need a GCP account to deploy model on Google Cloud Platform.


