# COVID-19_Prediction

In their paper in the BMJ Supportive & Palliative Care in July 2020 [1], Parchure and colleagues propose a model to detect mortality in patients with COVID-19 infection based on multiple features including patient demographics, laboratory test results and vital parameters.
For every patient, daily feature vectors daily feature vectors starting from admission date until the date of discharge or death were build.
For these feature vectors, the three most recent recorded assessments from time-series data that were available when each feature vector was created, were used.

## Labeling strategy
The authors write the following about their labeling srategy:

"The interval between the time of discharge and the time of generating each feature vector was generated daily for each patient. If the discharge disposition was ‘Expired (ie, dead)’ and the interval was between 20 and 84 hours, we labelled the feature vectors as positive. If the discharge disposition was ‘Not Expired’ and the interval was between 20 and 84 hours, we labelled the feature vectors negative. We excluded the remaining feature vectors from our cohort."





1. Parchure P, Joshi H, Dharmarajan K, Freeman R, Reich DL, Mazumdar M, et al. Development and validation of a machine learning-based prediction model for near-term in-hospital mortality among patients with COVID-19. BMJ Support Palliat Care. 2020 Sep; 
