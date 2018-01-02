# Enron POI Machine Learning Project

#### Project objective, data exploration, outlier investigation

The goal of this project is to predict whether an Enron employee is considered to be a Person of Interest (POI) in the corporate fraud case. In this dataset, we use a combination of employee financial data and email-related data to train a supervised learning model. Given the binary nature of our target variable (0 = not a person of interest, 1 = is a person of interest), then this is a classification problem.

The dataset has 146 observations. Looking at the target variable, 18 observations (12.3%) are classified as POI (class=1), while 128 are classified as not-POI (class=0). This translates to a pretty imbalanced dataset.

In terms of the explanatory variables, the dataset had a total of 20. On financial data, the variables in the dataset are:

* 'salary'
* 'deferral_payments'
* 'total_payments'
* 'loan_advances'
* 'bonus'
* 'restricted_stock_deferred'
* 'deferred_income'
* 'total_stock_value'
* 'expenses'
* 'exercised_stock_options'
* 'other'
* 'long_term_incentive'
* 'shared_receipt_with_poi'
* 'restricted_stock'
* 'director_fees'

On email-related data, the variables in the dataset are:

* 'to_messages'
* 'email_address'
* 'from_poi_to_this_person'
* 'from_messages'
* 'from_this_person_to_poi'

Most variables have missing values, denoted as 'NaN' in the dataset. Using pandas, I tabulated the number of missing values per variable. Here's the result:

* salary: 51
* to_messages: 60
* deferral_payments: 107
* total_payments: 21
* loan_advances: 142
* bonus: 64
* restricted_stock_deferred: 128
* deferred_income: 97
* total_stock_value: 20
* expenses: 51
* from_poi_to_this_person: 60
* exercised_stock_options: 44
* from_messages: 60
* other: 53
* from_this_person_to_poi: 60
* long_term_incentive: 80
* shared_receipt_with_poi: 60
* restricted_stock: 36
* director_fees: 129
* poi: 0

Dealing with outliers is a tricky endeavor in this dataset. Extremely large values could indicate an error in the dataset (in which case we should remove the data point), or useful signals for our classifier (ie, very large expenses could be indicative of POIs). After exploring the dataset a bit, I found that the highest salary did not belong to a person; instead, it mapped to 'TOTAL'. Since 'TOTAL' is not a person, this is likely a data input error that would throw off our results. Since the dataset is quite small, it's important to make sure that the data is as clean as possible. In this case, I handled the outlier by removing the 'TOTAL' instance from the dataset.

#### Feature selection

In my final model, I ended up using 4 independent variables. They are:

* expenses
* exercised_stock_options
* other
* from_ratio (this one I engineered myself)

3 of the 4 variables are related to financial data, while the last variable has to do with email data. Adding any other variable would reduce the overall performance of my classifier.

My selection process was fairly straight forward: I started with all the variables in the dataset, ran an ML model, and looked at the evaluation metrics (specifically, accuracy, precision, and recall). From there, I removed a single variable and reran the model. If the scores went down after removing a particular variable, then this was indicative that this variable had predictive power and should be used in my final dataset. I recursively repeated this process, pruning down my dataset until I only had a set of features that enhanced the performance of my final model.

I also engineered 3 features: the payments_salary_ratio, the from_ratio, and the to_ratio. The payments_salary_ratio is a ratio that looks at the total_payments for an employee and divides it by the salary of that employee. My rationale for using this feature was that some POI may not necessarily have extremely high total_payments relative to others because they could be more junior, so it could be helpful to compare it to their base salary. Perhaps people who have a total_payments significantly higher than their salary could provide a signal vis-a-vis suspicious behavior. This feature, however, did not really add to the predictive power of my final model and therefore was not used. The other 2 features - the from_ratio and the to_ratio - had a similar rationale. Rather than looking at the absolute volume of emails to and from a POI, it would be more helpful to normalize that volume by the total number of emails to and from this person. Some people, such as an administrative assistant, likely get/send a high volume of emails, and will have emails to a from a POI just by random chance. Normalizing this number could therefore add some explanatory power. While the to_ratio (from_poi_to_this_person / to_message) did not enhance predictive performance, the from ratio (from_this_person_to_poi / from_messages) did add predictive power and was consequently incorporated in the final model.

Lastly, I used sklearn's MinMaxScaler to scale my features between 0 and 1. Some ML algorithm's I implemented, such a K-Nearest Neighbors, only perform as intended if the features are standardized. In KNN's case, more weight would be given to a feature with large numbers, such as salary, than a feature with smaller numbers, such as the emails from a POI. With decision trees and ensembles of decision trees (Random Forests, Boosted Trees), scaling features doesn't really enhance or detract model performance (I tested this). To keep things simple, I ran all my final models with min-max scaled features.

#### Model selection

During this project, I tested the following ML algorithms using the sklearn library:

* Gaussian Naive Bayes classifier
* Decision Tree classifier
* Random Forest classifier
* AdaBoost classifier (selected model)
* Gradient Boosting classifier
* K-Nearest Neighbors

I ended up using the AdaBoost classifier, as this model yielded the best overall performance with an accuracy of 87.8%, a precision of 62.4%, and a recall of 38.2%. The Naive Bayes classifier had an accuracy score of 84.9%, precision of 43%, and recall of 18.4%. The Decision Tree yielded an accuracy of 85%, a precision of 30% but only a 3% recall. The Random Forest classifier yielded 85% accuracy, 48% precision, and 21% recall. The Gradient Boosting classifier passed the requirements of this project, with an accuracy of 85.8%, a precision of 50%, and a recall of 35.6%. However, it still underperformed relative to the AdaBoost classifier. Lastly, I ran a KNN classifier: this yielded an accuracy of 86.7% and the highest precision score at 67.4%. However, the recall significantly underperformed the ensembles with a score of 14.5%.  

#### Parameter tuning

Some machine learning models have parameters that cannot be learned from the data, but rather are set prior to training. These hyperparameters are tuned - or changed - much like radio frequency knobs, to optimize the performance of the model given a specific dataset. If a data scientist/analyst does not tune the parameters carefully, the she/he runs the risk of over/underfitting the data. As such, the model will underperform when it comes time to predict new data.

Different algorithms have different parameters that we can tune. In the case of K-Nearest Neighbors, the most important parameter is "K", which denotes the number of neighboring observations to compare a new data point to then make a classification. When K is very large, then the model overgeneralizes, simply taking an average of all the data points. When K is 1, the model overfits to whatever the nearest datapoint (vector with the smallest distance) looks like. With this project, the best K value was 5, after which the model began to underperform.

For the AdaBoost classifier, there were several knobs I ended up tuning. The first parameter I tuned was n_estimators, which is the "max number of estimators at which boosting is terminated." (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) The default number is 50, but I ended up using 8 as this was the best performing. I then tuned the learning rate, which shrinks the contribution of each subsequent classifier in the ensemble. The default was 1, but I ended up using a learning rate of 0.7. Lastly, I tuned the random state, which is the seed used by the number generator. Although the default value is None, which is essentially np.random, in my case I used a random_state = 1. This parameter, however, does not have a material impact on model performance. In this project, the most important hyperparameter when it came to model evaluation was the n_estimators. Although I did not use grid search, I systematically tested parameters - both higher and lower than the default values - until I converged at the best performing model. Specifically, I tested n_estimators= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 75, 100] and learning_rate = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.5].

#### Validation

One of the biggest challenges in data science and predictive modeling is managing the bias-variance tradeoff. Predictive models often overfit the data - in other words, they generate very complex models that do not generalize well at the time of prediction. To ensure we are not overfitting the data, we deliberately hold out a chunk of our dataset when training the model. We can then use the untrained chunk of our dataset to create predictions on the trained model, and compare those predictions vs the actual outcome. The evaluation metrics then provide a guidepost as to how the model predicts 'unseen' data.

Given the limited size of the dataset, holding out on some data means that we have very few data points to train the model on. Using sklearn, we can implement cross validation, which generates several models on different cross sections (or 'folds') of the data, and then takes an average performance of the predictions. This enables us to get a lot more bang for our limited data. Specifically in this project, due to the imbalanced nature of our target variable, we are using a stratified shuffle split, which randomizes the number of splits and "ensures that the relative class frequencies is approximately preserved in each train and validation fold." (http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

#### Evaluation

For classification models, the accuracy score denotes the total instances predicted correctly / total observations in the dataset. However, accuracy alone doesn't always give us the full picture. In classification, we also care about the relevance of our model's prediction, which is measured via precision and recall. Precision is calculated as the total number of true positives / total positive predictions. In other words, if our algorithm predicts that an employee is a POI, what are the chances that the employee is actually a POI. The recall is calculated as the true positives / total number of actual positive instances. In other words, given a POI in our dataset, what are the chances that we classified him/her as one?

In this project, the AdaBoost classifier yielded an accuracy of 87.8%, a precision of 62.4%, and a recall of 38.2%. Since the precision is considerably higher than the recall, this classifier is pretty picky and does not classify too many employees as POIs. When the classifier predicts someone is a POI, it is pretty likely that he/she is in fact a POI. However, there are often POIs that get classified as not being POIs. Given the high legal fees, social stigma, and reputational costs associated with being labeled a person-of-interest in the largest corporate fraud case in America, to me it is more socially palatable to have a classifier that errs on the side of not classifying a POI than over-classifying everybody as a POI.
