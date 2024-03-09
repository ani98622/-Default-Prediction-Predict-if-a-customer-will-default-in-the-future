# Default Probability Prediction
**Note** :  For `EDA` and `Model Selection` files visit `notebook` folder directly

## **Objective**
Predict the probability of a customer defaulting on their credit card balance based on their monthly profile. 

## **Data Sets**
- **test_data.feather (2.74 GB)**: Test data for predicting the target label for each 'customer_ID'.
- **train_data.feather (1.33 GB)**: Training data with multiple statement dates per 'customer_ID'.
- **train_labels.csv (29.3 MB)**: Target 'label' for each 'customer_ID'.

## **Variable Description**
- **D_***: Delinquency variables
- **S_***: Spend variables
- **P_***: Payment variables
- **B_***: Balance variables
- **R_***: Risk variables

Categorical predictors: ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'].

## **Output Format**
- CSV file with two columns:
  - Customer_ID
  - Probability

## **References**
- Delve into the concept of [Weight of Evidence (WoE) and Information Value (IV)](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html).
- Explore advanced binning strategies with [Optbinning](https://gnpalencia.org/optbinning/).
- Convert .csv to .feather format effortlessly using this [Kaggle Dataset](https://www.kaggle.com/datasets/seefun/amex-default-prediction-feather).
