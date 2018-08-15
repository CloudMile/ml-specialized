# ML Specialized Project

CloudMile project for google ml-specialist 

## DNNRegressor/DNNLinearCombinedRegressor on Rossmann data 

Rossmann Data是與時間序列有關的Data, 需要預測每間商店未來六個禮拜的銷售額, 最大的挑戰是您的Model是否能夠有能力去抓到個別商店的平均銷售額以及週期性, 季節性的Information

- From kaggle [Rossmann](https://www.kaggle.com/c/rossmann-store-sales)
- See the [rossman.ipynb](rossmann/rossman.ipynb) for details

### Custom tf.estimator.Estimator for KKBOX Music Recommendation Engine

- Music Personalize Recommendation, here because of the target column is binary, we take this as a classification problem, in a nutshell, this is a binary classification.
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
        
將data放置到個別的專案底下, 並且照著notebook的流程執行(但是Deploy model到GCP cloud的部分就需要GCP的帳號)


