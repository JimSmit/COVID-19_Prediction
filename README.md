# COVID-19 Early Warning

## Introduction
In their paper in the BMJ Supportive & Palliative Care in July 2020 [1], Parchure and colleagues propose a model to detect mortality in patients with COVID-19 infection based on multiple features including patient demographics, laboratory test results and vital parameters. 
The primary outcome of interest was in- hospital death within 20–84 hours from the time of prediction.
Another paper from this research group [2] did a similar study, but predict ICU transfer in COVID patients within 24 hours. 

For every patient, they sampled **daily** feature vectors daily feature vectors starting from admission date until the date of discharge or the event (death/ICU transfer) were build.
For these feature vectors, the three most recent recorded assessments from time-series data that were available when each feature vector was created, were used.

<img src="https://raw.githubusercontent.com/JimSmit/COVID-19_Prediction/main/images/pos_label.PNG" width="300">
<img src="https://raw.githubusercontent.com/JimSmit/COVID-19_Prediction/main/images/neg_label.PNG" width="300">

Feature vectors were only labeled positive if the 'event' (patient's death is this case) occured somewhere between 20 and 84 hours after the timestamp of the feature vector (the time of prediction). That means, this sampling strategy forces a model to predict a patient death in a prediction window of 64 hours, with a gap of 20 hours between the moment of prediction and the start of the prediction window. 

**Feature window**: 3 most recent accessments of variable (so actual window size depends on variable availability).
**gap**: 20 hours.
**prediction window**: 64 hours.

<img src="https://raw.githubusercontent.com/JimSmit/COVID-19_Prediction/main/images/windows.png" width="500">

This work presents a similar model for ICU transfer prediction within 24 hours, based on the data from the Erasmus Medical Center in Rotterdam.
It was inspired by the 2 mentioned studies, but differs from it in a couple of ways:

1. sampling / labeling strategy:
- As the We only use the most recent accessment.
- We every patient daily, and use all samples for model training and validation.
- We label as follows:  If the time between sampling and the event (ICU transfer), aka the 'time-to-event', is 
(1) shorter then the prediction window, but larger than the gap, we labeled the sample as **positive**;
(2) larger than the prediction window + gap, we labeled the sample as **negative**;
(3) shorter than the gap, we labeled the sample as **negative**.

<img src="https://raw.githubusercontent.com/JimSmit/COVID-19_Prediction/main/images/sampling_strategy.png" width="800">

2. Used features:
To include information about vital parameter trends, we added extra features using a moving feature window, including several statistics of the vital parameters measured withing the feature window (max, min, std, diff, mean, median)

3. ML models
We try different types of models, including simple linear models and more complex non-linear models.

4. Imputation
Instead of median imputation, we train an imputer algorithm based on the training set. Scikitklearn's iterative imputer is used, which is simimlar to the R package MICE.

5. Model validation by Cross-validation:
A split is made to perform internal validation. The model is trained and optimized without any knowledge from the left-out validation set.

6. Model interpretability:
To increase interpretability of the risks calculated by the model, patient- and time-specific feature importances are calculated by SHAP [3].



Literature:
1. Parchure P, Joshi H, Dharmarajan K, Freeman R, Reich DL, Mazumdar M, et al. Development and validation of a machine learning-based prediction model for near-term in-hospital mortality among patients with COVID-19. BMJ Support Palliat Care. 2020 Sep; 
2. Cheng FY, Joshi H, Tandon P, Freeman R, Reich DL, Mazumdar M, et al. Using machine learning to predict ICU transfer in hospitalized COVID-19 patients. J Clin Med [Internet]. 2020;9(6). Available from: https://www.embase.com/search/results?subaction=viewrecord&id=L2004497780&from=export
3. Lundberg SM, Lee S. A Unified Approach to Interpreting Model Predictions. 2017;(Section 2):1–10. 
