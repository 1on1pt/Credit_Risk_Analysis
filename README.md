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
The **performance** of a machine learning algorithm can be assessed by the model's:
* <ins>Balanced Accuracy Score</ins>:  Compares actual outcome (y_test) with the predicted outcome (y_pred) for a percentage of predictions that are correct.
* <ins>Precision Score</ins>:  A measure of how reliable a positive classification is and can be determined by dividing the number of true positives (TP) by the number of all positives (true positives (TP) + false positives (FP)).  This formula can be used:  Precision = TP / TP + FP.
* <ins>Sensitivity or Recall Score</ins>:  Sensitivity or recall is a measure of the proportion of actual positive cases which got predicted as *positive (or true positive)*.  This formula can be used for sensitivity/recall:  Sensitivity = TP / TP + FN.

### Oversampling Review
#### RandomOverSampler Model
* Balanced Accuracy Score = 65.7%

![ros_balanced_accuracy](https://user-images.githubusercontent.com/94148420/163693170-e76219e9-0e06-4d9e-8d2b-ba77feacd329.PNG)

* Precision Score = 1%
* Recall Score = 71%

![ros_class_report](https://user-images.githubusercontent.com/94148420/163693201-a757479c-6a67-43cf-9200-fc80ab211ced.PNG)

#### SMOTE Model
* Balanced Accuracy Score = 66.2%

![SMOTE_balanced_accuracy](https://user-images.githubusercontent.com/94148420/163693249-7e40fc64-7f1d-447c-b8a8-4338970f6079.PNG)

* Precision Score = 1%
* Recall Score = 63%

![SMOTE_class_report](https://user-images.githubusercontent.com/94148420/163693268-4eb464d0-1b1a-4d5a-aa65-fb6ad4f5f519.PNG)


### Undersampling Review
#### ClusterCentoids Model
* Balanced Accuracy Score = 54.4%

![ClusCent_balanced_accuracy](https://user-images.githubusercontent.com/94148420/163693330-da6af9be-7663-44e5-9aa4-758f72a66496.PNG)

* Precision Score = 1%
* Recall Score = 69%

![ClusCent_class_report](https://user-images.githubusercontent.com/94148420/163693360-094f91cc-d298-4f30-a1fa-97111896b64a.PNG)


### Combination (Over and Under) Sampling Review
#### SMOTEENN Model
* Balanced Accuracy Score = 68.8%

![EENN_balanced_accuracy](https://user-images.githubusercontent.com/94148420/163693424-9d997200-47fd-4d78-a315-ccf56ebe4a74.PNG)

* Precision Score = 1%
* Recall Score = 80%

![EENN_class_report](https://user-images.githubusercontent.com/94148420/163693438-6d4fb79b-04a6-420c-9d6e-2e27889658b2.PNG)


### Ensemble Classifiers Review
#### BalancedRandomForestClassifier Model
* Balanced Accuracy Score = 78.8%

![BRFC_balanced_accuracy](https://user-images.githubusercontent.com/94148420/163693510-be9d9ded-3d9f-4117-83f9-c0893e5a0bee.PNG)

* Precision Score = 3%
* Recall Score = 70%

![BRFC_class_report](https://user-images.githubusercontent.com/94148420/163693550-9ed15329-ca32-4b75-8949-f33579eb4b45.PNG)

#### EasyEnsembleClassifier Model
* Balanced Accuracy Score = 93.2%

![EEC_balanced_accuracy](https://user-images.githubusercontent.com/94148420/163693601-e8a05ba4-0142-4a16-bfe0-6832662e4a7a.PNG)

* Precision Score = 9%
* Recall Score = 92%

![EEC_class_report](https://user-images.githubusercontent.com/94148420/163693620-64d21083-c810-4bd2-8826-f1424b179a61.PNG)



## Summary

