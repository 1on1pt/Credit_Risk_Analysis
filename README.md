# Credit_Risk_Analysis
Using machine learning algorithms from scikit-learn and imbalanced-learn to identify credit risk from a LendingClub dataset.

## Overview of the Analysis
![LendingClub_logo](https://user-images.githubusercontent.com/94148420/163691282-bccf8b63-cef0-4849-b761-13c313174085.jpg)

Jill commends you for all your hard work. Piece by piece, you’ve been building up your skills in data preparation, statistical reasoning, and machine learning. You are now ready to apply machine learning to solve a real-world challenge: **credit card risk**.

Credit risk is an inherently *unbalanced classification problem*, as good loans easily outnumber risky loans. Thus there is a need to employ different techniques to train and evaluate models with unbalanced classes. Jill asks you to use **imbalanced-learn** and **scikit-learn libraries** to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, oversampling the data was performed using the following algorithms:
* RandomOverSampler
* SMOTE

And undersampling the data using this algorithm:
* ClusterCentroids algorithm

Then next, a combinatorial approach of over- and undersampling is utilized using this algorithm:
* SMOTEENN

Next, there is a comparisson of two new machine learning models that reduce bias in predicting credit risk:
* BalancedRandomForestClassifier
* EasyEnsembleClassifier

Finally, an evaluation of the performance of these models with a recommendation on whether they should be used to predict credit risk.


### Resources
#### Code:
* credit_risk_resampling.ipynb
* credit_risk_ensemble.ipynb

#### Data:
* https://github.com/1on1pt/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.csv

#### Software/Data Tools/Algorithms:
* Jupyter Notebook 6.4.6
* Python 3.7.11
* scikit-learn
* imbalanced-learn
* RandomOverSampler
* SMOTE
* ClusterCentroids
* SMOTEENN
* BalancedRandomForestClassifier
* EasyEnsembleClassifier


## Results
The performance of a machine learning algorithm can be assessed by the model's:
* <ins>Balanced Accuracy Score</ins>:  Compares actual outcome (y_test) with the predicted outcome (y_pred) for a percentage of predictions that are correct.
* Precision Score:  A measure of how reliable a positive classification is and can be determined by dividing the number of true positives (TP) by the number of all positives (true positives (TP) + false positives (FP)).  This formula can be used:  Precision = TP / TP + FP.
* Sensitivity or Recall Score






## Summary

