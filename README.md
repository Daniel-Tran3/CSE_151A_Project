## Project Objective
In this project, we will be exploring an apartment rental [dataset](https://github.com/Daniel-Tran3/CSE_151A_Project/blob/main/apartments_for_rent_classified_10K_utf.csv) and building a model to predict rental prices.

### Preprocessing Approach
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

### (See Milestone 3 branch README for Model 1 Summary)
Colab link to the first model: <a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model_1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Model 2 - Logistic Classification
Colab link to second model:<a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model_2_Pre.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Evaluation of data, labels and loss function.**\
As we decided to co-opt our first model, the ANN that predicts rental prices given a dataset, we did some additional preprocessing.
We began by using our model to give predicted prices, assumed to be the baseline for "fairness" as an aggregate over many data points.
We then compared these prices to the original listed prices, assigning the original listed prices as "unfair" if they were 1.3x
more expensive than our model's predictions or "fair" otherwise - our two new labels.
Finally, we updated our loss function to be binary cross entropy, since we were performing binary classification.

**Training vs test error / fitting graph**\
As our second model was a binary classification model instead of a regression model, we switched our measurement of error from
MSE to precision and recall.
After using 10-fold cross-validation (explained in further detail below), we found that precision averaged 
0.92 and recall averaged 0.95 on the training data, while precision and recall both averaged 0.8 on the training set.
(More below, under "Fitting graph")

**Fitting graph**\
Here are some graphs of the model's predictions versus the actual data:

<Insert graphs here>

Compared to our previous model's accuracy, this model seemed to weaken a bit due to overfitting.
While the testing results weren't entirely inaccurate, it is clear from the much-higher training accuracy that 
our model is too specialized to the training data. This is one aspect that we seek to improve for our next classification model.

**Hyperparameter tuning and K-fold Cross validation**\
We decided to perform hyperparameter tuning and 10-fold cross validation, as mentioned above.
The hyper-parameter tuning allowed us to test multiple activation functions, units, and optimizers.
The results of the hyper-parameter tuning gave the following optimal hyperparameters:
12 units per hidden layer, Adadelta optimizer, 0.9 learning rate, and relu activation functions.

The relu activation function is of particular interest, as it aligns with what we found in Model 1 and further 
confirms that the relationship between the majority of our features and predicted price is linear.

We also decided to include cross-validation to test for over/underfitting, and (as previously mentioned), 
overfitting was indeed present.

**Next model**\
Support Vector Machine: Like the logistic classification model, we also intend to use a support vector machine to classify prices as fair and unfair. We want to see if an SVM will have less overfitting than the logistic classification neural network.
We will keep the above-described method to create and add "unfair" entries to the table. The support vector machine will then create a separating margin between the "fair" and "unfair" classes.

**Conclusion**\
The second model used the results of the first model to label each entry as a "fair" or "unfair" price, assuming that our model
correctly predicts a "fair" price for an apartment by using the aggregate of all of the initial data. It then uses
hyperparameter tuning to obtain a logistic classification model with the best parameters across number of units, optimizer,
learning rate, and activation functions (specific valud above, in "Hyperparameter tuning and K-fold Cross validation"). 

We then used 10-fold cross validation to check for overfitting, which was unfortunately present in our model. Despite this, our model
still achieves passable precision and recall on testing data. However, we wish to improve these values further in our next SVM model.

---
Our project can be found here: 
<a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/CSE151A_Group_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
