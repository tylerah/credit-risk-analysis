# Analysis of Credit Risk Using Supervised Machine Learning
## Overview
Personal loans are one of the most lucrative services provided by banks and various lenders. However, it is important that a thorough analysis of each potential borrower is performed before issuing a loan in order to ensure that the money will be repaid. In this project, various supervised machine learning algorithms were performed in order to automate this evluation process. The accuracy and precision of each process was evaluated in order to make a recommendation as to which would perform the best. 

The following types of algorithms were evaluated:
1. Naive Random Oversampling
2. SMOTE Oversampling
3. Cluster Centroid Undersampling
4. SMOTEENN Sampling
5. Balanced Random Forest Classifying
6. Easy Ensemble Classifying

Note that the first four methods were run in a jupyter notebook while the last two (Balanced Random Forest and Easy Ensemble) were run in a Google Colaboratory notebook. The results of each are provided in the following section. 

## Results
### Naive Random Oversampling
In the random oversampling method, instances of the minority class are selected at random. They are then added to the training set until there is a balance between majority and minority classes. 

The following image displays the results of this algorithm:

![naive_random_oversampling](https://user-images.githubusercontent.com/104606662/187839616-c521580c-25ef-4ae9-b824-f67d046f8be0.png)

Results:
* Balanced Accuracy: 64.03%
* Precision High Risk: 1%
* Precision Low Risk: 100%
* Recall High Risk: 66%
* Recall Low Risk: 62%

### SMOTE Oversampling
SMOTE (synthetic minority oversampling technique), is similar to the random oversampling method. However, in this case the new instances added are interpolated. For example, when a new instance is selected from the minority class a number of that instance's closest neighbors are also selected. Based on these values, new values are created. 

The following image displays the results of this algorithm:

![SMOTE_oversampling](https://user-images.githubusercontent.com/104606662/187839637-7ea1642c-df82-45c1-8c48-3e8a356cc70d.png)

Results:
* Balanced Accuracy: 65.15%
* Precision High Risk: 1%
* Precision Low Risk: 100%
* Recall High Risk: 71%
* Recall Low Risk: 57%

### Cluster Centroid Undersampling
This method is somewhat similar to the SMOTE method. However, instead of interpolating neighbors of the minority class, this method identifies clusters of the majority class and then generates sythetic datapoints that are representative of the cluster. These clusters are known as centroids. After the centroids are generated, the majority class is undersampled to match the size of the minority class. 

The following image displays the results of this algorithm:

![cluster_centroid_undersampling](https://user-images.githubusercontent.com/104606662/187839564-eb56b45d-2593-4af8-9519-d285c6602bdc.png)

Results:
* Balanced Accuracy: 54.47%
* Precision High Risk: 1%
* Precision Low Risk: 100%
* Recall High Risk: 69%
* Recall Low Risk: 40%

### SMOTEENN Sampling
SMOTEENN (SMOTE and Edited Nearest Neighbors) is an approach to resampling that combines aspects of oversampling and undersampling. It involves two steps. The first step is to oversample the minority class using SMOTE (see above description). Next, it cleans the resulting data via undersampling. In this case if two nearest neighbors of a data point belong to two difference classes, that data point is dropped. 

The following image displays the results of this algorithm:

![SMOTEENN_sampling](https://user-images.githubusercontent.com/104606662/187839690-a420e955-d047-43c5-abea-2df918456e0c.png)

Results:
* Balanced Accuracy: 64.29%
* Precision High Risk: 1%
* Precision Low Risk: 100%
* Recall High Risk: 71%
* Recall Low Risk: 57%

### Balanced Random Forest Classifying
This method is an example of an ensemble classifier. Ensemble learning is a process of combining multiple models to help improve accuracy and robustness of the machine learning model. 

Balanced Random Forest Classifying is a modification of Random Forest Sampling. In this method, two bootstrapped sets of the same size are generated from each tree. These sets are equal in size to the minority class and one is used for the minority class while the other is used for the majority class. These two sets constitute a training set. 

![balanced_random_forest](https://user-images.githubusercontent.com/104606662/187839532-8c1f7743-a347-412b-b0fe-526e0ce8e59f.png)

Results:
* Balanced Accuracy: 78.85%
* Precision High Risk: 3%
* Precision Low Risk: 100%
* Recall High Risk: 70%
* Recall Low Risk: 87%

### Easy Ensemble Classifying
This method is also known as Adaptive Boosting. In this model, the model is trained and then subsequently evaluated (in contrast to other methods). After errors of the first model are evaluated, a second model is trained. During the second training sequence, errors are weighted to minimize the previous errors. This process is repeated until the error rate is minimized. 

The following image displays the results of this algorithm:

![easy_ensemble_classifying](https://user-images.githubusercontent.com/104606662/187839585-22bf1886-9d41-4247-bd40-c74843592b79.png)

Results:
* Balanced Accuracy: 99.17%
* Precision High Risk: 12%
* Precision Low Risk: 100%
* Recall High Risk: 100%
* Recall Low Risk: 98%

## Summary and Recommendation
In order to recommend the best model, it is important to understand the practical context of what the model is being used for. In this case, lenders want to successfully predict which borrowers are high risk and which are low risk. Specifically, lenders want to prevent as many high risk borrowers from passing through the screen while maximizing as many low risk borrowers as possible. In other words, the model needs to be sensitive / have a high recall for high-risk borrowers. As such, it is recommended that the Easy Ensemble / Adaptive Boosting be used as this model had the highest balanced accuracy, the highest precision for high risk borrowers, and the highest recall for high risk borrowers. 
