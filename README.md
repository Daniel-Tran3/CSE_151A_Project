## Introduction
For our project, we are using a dataset that contains apartment rental information across the country, and it included important features such as the price, amenities, location, and number of rooms. It was chosen for the plethora of features and observations (10,000 rows) available to us, and it provided a glimpse into the housing rental market of America. Our project has three main models: the first model predicts the rental price, and we compare the predicted price with the real price to give the listing a label (fair or unfair); the second model aims to predict that label using logistic regression, and the third model aims to predict the same label using SVM. The goal for the project is to identify and predict the fairness of the price, and this is important because if our model is deployed in the real-world, it will be able to provide millions of people looking for housing crucial information on the price. With the fairness of price in hand, consumers can make a more informed decision and avoid the unfair pricing, which will eventually drive those prices down to a more reasonable level.


## Methods

### Project Objective
In this project, we will be exploring an apartment rental [dataset](https://github.com/Daniel-Tran3/CSE_151A_Project/blob/main/apartments_for_rent_classified_10K_utf.csv) and building a model to predict rental prices.

### Data Exploration

### Preprocessing Approach (Change as needed)
We want to ensure the integrity of our data, in order to achieve a high level of accuracy when modelling. These are some of the steps we have taken:

**Handling Missing Values**\
We addressed missing values by filling categorical fields like amenities and pets_allowed with "None" or a similar placeholder. For numeric fields such as bathrooms and bedrooms, we used the median to fill in the gaps, preserving the data distribution.

**Feature Engineering**\
We transformed the amenities field into multiple binary columns, each indicating the presence or absence of specific amenities. This approach allows a more detailed analysis of amenities' effects on rentals.

**Encoding Categorical Variables**\
In order to simplify our dataset, categorical variables including currency and price_type have be converted to one-hot encoded vectors, eliminating the introduction of arbitrary numeric relationships. This ensures a clearer relationship and conclusion.

**Normalization**\
Numeric fields like price and square_feet will be normalized to ensure uniformity in scale across our dataset, which is crucial for our models' performance.

**Outlier Management**\
We carefully examine price and square_feet for outliers, choosing to cap or remove them based on their severity and impact on the dataset. 

**Text Data Preprocessing**\
For textual data in title and body, we plan to clean up by removing special characters and standardizing text cases, setting the stage for potential text analysis and feature extraction.

**Data Types Correction**\
Ensuring all columns are of the correct data type is a priority, including accurately converting boolean fields and validating the consistency of numerical fields.

This preprocessing strategy ensures our dataset is primed for analysis and detailed exploration (including modeling), focusing on cleanliness, structure, and readiness for in-depth exploration. It lays the groundwork for robust and insightful data-driven models and findings. 

---

### Model 1

### Model 2

### Model 3


## Results

### First Model
Colab link to the first model: <a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model_1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>




### Second Model
Colab link to the second model: <a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model_2_Pre.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Training vs test error / fitting graph**\
As our second model was a binary classification model instead of a regression model, we switched our measurement of error from MSE to precision and recall.
After using 10-fold cross-validation (explained in further detail below), we found that precision averaged 
0.87 and recall averaged 0.95 on the training data, while precision averaged 0.77 and recall averaged 0.80 on the training set. Additionally, accuracy averaged 0.85 on the training set and 0.7 on the testing set.

It should be noted that the cross-validation iteration graphs were slightly different than the training, testing, and validation (not included in K-cross-fold) results from predicting once on each set of data.

These predictions gave precisions of 0.75 and recalls of 0.99 for class 1.
For class 0, however, the precisions were 0.66, 0.44, and 0.58 (for training, testing, and validation). The recalls were even worse, at 0.07, 0.04, and 0.07. 
Clearly, our model has a heavy bias towards classifying apartments as "fair".
This tendency, and possible explanations, will be discussed later.

**Fitting graph**\
Here are some graphs of the model's training vs. testing metrics for the K-fold cross-validation:

![image](https://github.com/Daniel-Tran3/CSE_151A_Project/assets/62851286/17713800-f621-42da-b561-2d95fbfe1425)

![image](https://github.com/Daniel-Tran3/CSE_151A_Project/assets/62851286/8e3f8754-cc45-4ad8-84e8-4320c007f34a)

![image](https://github.com/Daniel-Tran3/CSE_151A_Project/assets/62851286/ef40d580-886c-4c29-b042-440331c60db1)

**Hyperparameter tuning and K-fold Cross validation**\
We decided to perform hyperparameter tuning and 10-fold cross validation, as mentioned above.
The hyper-parameter tuning allowed us to test multiple activation functions, units, and optimizers.
The results of the hyper-parameter tuning gave the following optimal hyperparameters:
18 units per hidden layer, Adadelta optimizer, 0.9 learning rate, and tanh activation functions.

One interesting feature we noticed while running the testing was that the accuracy did not seem 
to vary beyond 0.74 and 0.76 for the entire search - almost as if the actual choices for the 
above hyperparameters ultimately did not increase the accuracy of the model significantly.



### Third Model
Colab link to the third model: <a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model3.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>




### Discussion





### Conclusion




Our project can be found here: 
<a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/CSE151A_Group_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
