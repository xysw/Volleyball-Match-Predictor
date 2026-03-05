# Volleyball-Match-Predictor

A machine learning project that predicts the outcome of beach volleyball matches by training an XGBoost algorithm (gradient boosted decision trees) on competition data!

The dataset used in this project contained player and match info from FIVB and AVP volleyball competitions, and can be found here: https://github.com/BigTimeStats/beach-volleyball

This project was created to become familiar with the machine learning process and learn how to visualise and analyse data, engineer features, handle missing values, perform hyperparameter tuning, and understand the model training process. 

## Technologies
* Python 3.12.10


## Launch
### Run locally:
1. Download the dataset, vb_matches.csv, and place it in the same directory as the notebook.

2. Install dependencies
```
pip install pandas numpy matplotlib scikit-learn xgboost hyperopt
```
3. Run all cells in app.ipynb

## Process
1. Investigated distributions between key features in the data, including age, height, and ranking. Determined that age and height distributions are identical between winning and losing teams, while ranks significantly affect the target column of winning a match. Match statistics contained many missing values.

2. Preprocessed data - converted rankings from a mixed format into a main rank and qualifier rank, converted max duration to minutes, imputed missing age using median grouped by gender and circuit, imputed missing height values using median grouped by gender and age bins, frequency encoded tournament and country features, binary encoded gender and circuit features, and manually encoded round and bracket features.

3. Feature engineering - created a home-country advantage feature, combined game statistics data into an error_ratio feature, and restructured data from "winning team" and "losing team" format to "Team A" and "Team B", with a target column of "Team A wins". Flipped 50% of rows to balance data.

4. Established a naive baseline algorithm predicting entirely on ranks, and also trained a baseline logistic regression algorithm, both reaching 70% accuracy and 0.78 AUROC.

5. Trained XGBoost with default hyperparameters and tuned using HyperOpt library (78.0% accuracy, 0.88 AUROC). Test accuracy matched validation accuracy. XGBoost outperformed the baseline algorithms by 8% accuracy and 0.10 AUROC.

## Limitations
* Huge portion of game statistics data missing limits the model's ability to consistently use these performance-based features
* A gap of 5% between training and validation dataset accuracy suggests the model may be overfitting
Hence, future exploration will be to prevent overfitting, maybe with more conservative hyperparameter restraints, or looking into handling missing game statistics data.
