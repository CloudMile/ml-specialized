# KKBOX Music Recommendation Model 

![kkbox_picture](./kkbox_picture.jpg)

## Overview

The 11th ACM International Conference on Web Search and Data Mining (WSDM 2018) is challenging you to build a better music recommendation system using a donated dataset from KKBOX. WSDM (pronounced "wisdom") is one of the the premier conferences on web inspired research involving search and data mining. They're committed to publishing original, high quality papers and presentations, with an emphasis on practical but principled novel models.

The ideal method for music recommendation would be analyzing binary music files in addition to analysis of the structured user logs. Nevertheless, due to the substantial size of binary files as well as the lack of such open data, here we focus on providing prediction of users' preference of songs from member history. The definition for preference can be found here Kaggle KKBOX Music Recommendation

We will build a standard DNN (Dense layer) model, also called fully connected layer, as well as the structure in Neural Collaborative Filteringand provide comparison. We also heavily use averaged embedding similar to the combiner function in tf.feature_column.embedding_column.


## What kind of problem to solve?

Click Throgh Rate (CTR) like or Classification problem


## Dataset

- The label of the dataset are in (0, 1), we colud treat this a binary classification problem.
- We put the dataset in `gs://ml-specialized/kkbox/data`.
- Detail description about dataset see [kkbox.ipynb](./kkbox.ipynb).


## EDA(Exploratory Data Analysis), Running Code, Model Structure, Training, Eval Tips

We put almost code in `.py` scripts rather than jupyter notebook, in notebook we demonstrate how to EDA, and how to run code 
step by step, see [kkbox.ipynb](./kkbox.ipynb) for detail.


