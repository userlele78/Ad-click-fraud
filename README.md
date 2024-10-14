# Ad-Click Fraud Detection Using Ensemble Learning
This repository contains the implementation and research conducted as part of the project titled "A Solution for Class Imbalance Problem in Ad-Click Fraud Detection using Ensemble Learning Models". The research compares the performance of several ensemble learning models to address class imbalance issues in detecting fraudulent ad-clicks.

**Project Organization**
```
├── README.md          <- The top-level README for developers using this project.
├── EDA                <- Jupyter notebooks for Exploratory Data Analysis.
├── K63_Thesis_AdClickFraud  <- Report on the project
└── src
    ├── feature engineering & modelling        <- Scripts for feature engineering & modelling in details
    └── helpers                                <- Scripts for data processing
        └── utils                              <- Scripts supporting for EDA 
```


**Introduction**
Ad-click fraud is a critical issue in online advertising that costs advertisers billions of dollars each year. Fraudulent clicks, performed either by humans or bots, inflate the performance of ads artificially, leading to financial losses. This project focuses on detecting fraudulent clicks using machine learning models, particularly tackling the problem of class imbalance.

We experiment with several state-of-the-art ensemble learning models such as XGBoost, LightGBM, and CatBoost, alongside oversampling techniques like SMOTE (Synthetic Minority Oversampling Technique) and feature selection approaches to improve fraud detection accuracy.

**Dataset**
The dataset used in this study was provided by [TalkingData](https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection) from the Kaggle competition, which contains click data from mobile advertisements. The key features include:

ip: IP address of the click
app: Application ID for the mobile app
device: Device type ID
os: Operating system version ID
channel: Advertising channel ID
click_time: Timestamp of the click
is_attributed: Target variable indicating whether the app was downloaded after the ad-click (1 if downloaded, 0 otherwise)

**Methodology**
The primary challenge in this project is class imbalance, where fraudulent clicks (positive class) make up a very small percentage of the dataset. Our approach combines data-level, algorithm-level, and ensemble learning techniques to tackle this issue. Specifically, we applied:

- SMOTE for oversampling the minority class.
- Feature Engineering using techniques such as interaction variables and time-based features.
- Hyperparameter tuning using Optuna to improve model performance.

**Ensemble Models**
We compare several model to make ensemble models:

XGBoost
LightGBM
CatBoost
Random Forest
AdaBoost
These models are evaluated for their ability to handle imbalanced data and detect fraudulent ad-clicks with high precision.

**Feature Engineering**
We engineered new features from the existing dataset to enhance model performance:
- Click count features: Count of clicks per combination of ip, device, os, app, and channel.
- Time-based features: Extracted day, hour, minute, and second from click_time.
- Interaction features: Features such as ip_device_os_nunique_app and ip_nunique_hour were created to capture user behavior patterns.
  
**Evaluation Metrics**
Given the class imbalance, traditional accuracy was not the primary metric. Instead, we used:
- F1-Score: Balances precision and recall, especially important for imbalanced datasets.
- AUC-ROC: Measures the model’s ability to distinguish between the positive and negative classes.
  
**Results**
The best model achieved an F1-score of 0.929 and an AUROC of 0.976 by using SMOTE, Linear Discriminant Analysis (LDA) for feature reduction, and CatBoost with hyperparameter tuning. The results suggest that the combination of oversampling, feature selection, and ensemble learning significantly improves the detection of fraudulent clicks.

**Conclusions**
CatBoost and LightGBM were the most effective models in handling imbalanced data, with CatBoost showing the best overall performance.
The integration of SMOTE with ensemble learning models proves to be a robust solution for class imbalance.
Feature engineering and hyperparameter tuning are crucial steps in maximizing the model’s potential.
