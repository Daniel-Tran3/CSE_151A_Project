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
Since we want to use our first model to make the distinction between fair and unfair monthly rents, we load that model along with out cleaned dataset. We take a subset of the data that doesn't include the price and time columns and make price predcitions for every apartment in the dataset. We create a new dataframe with these price predictions and ground truth prices. We made the choice to classify apartments with a listed monthly rent that is 30% more expensive than the predicted price as unfair (otherwise it is a fair price). These new labels are used to train the second model.

A random sample of 5 rows from this data without the hundreds of one-hot encoded features is shown below.

+------+-------------+------------+---------------+------------+------------------+
|      |   bathrooms |   bedrooms |   square_feet | fairness   |   unscaled_price |
+======+=============+============+===============+============+==================+
|  811 |           1 |          0 |    0.00949898 | fair       |              495 |
+------+-------------+------------+---------------+------------+------------------+
| 2191 |           1 |          1 |    0.013083   | fair       |             1145 |
+------+-------------+------------+---------------+------------+------------------+
| 6788 |           1 |          2 |    0.0225319  | fair       |             2200 |
+------+-------------+------------+---------------+------------+------------------+
| 5243 |           1 |          1 |    0.0181959  | fair       |             1191 |
+------+-------------+------------+---------------+------------+------------------+
| 1444 |           1 |          1 |    0.0112534  | fair       |              515 |
+------+-------------+------------+---------------+------------+------------------+

Using apartment features such as bedroom and bathroom counts, square footage, and location, we want to make a prediction on whether or not the apartment would be listed for a fair or unfair price. From our table with fairness labels, we created training, test, and validation sets. We used 80% of the data for training and took 20% of that to use as validation data.

Using Keras, we created a logistic regressor using 5 dense hidden layers. Each hidden layer, save for the first, had a variable number of nodes and possible activation functions that were tweaked during the hyperparameter tuning process. Additionally, we played around with different optimizers like Stochastic Gradient Descent, Adam, and Adadelta (a variation of SGD).

Here is the code snipper of how we constructed this model:

```

def call_existing_code(units, optimizer_candidate, activation, lr):
  model = keras.Sequential()
  model.add(layers.Dense(units=16, activation=activation, input_dim=X_train.shape[1]))
  model.add(layers.Dense(units=units, activation=activation))
  model.add(layers.Dense(units=units, activation=activation))
  model.add(layers.Dense(units=units, activation=activation))
  model.add(layers.Dense(units=units, activation=activation))
  model.add(layers.Dense(1, activation="sigmoid"))
  if (optimizer_candidate=="SGD"):
    optimizer=SGD(lr)
  elif (optimizer_candidate=="Adam"):
    optimizer=Adam(lr)
  elif (optimizer_candidate=="Adadelta"):
    optimizer=Adadelta(lr)
  else:
    optimizer=optimizer_candidate
  model.compile(
      optimizer=optimizer,
      loss="binary_crossentropy",
      metrics=["accuracy"],
  )
  return model

# Build model, using hyperparameters to choose options and use function above to build model
def build_model(hp):
  units = hp.Int("units", min_value=6, max_value=18, step=6)
  activation = hp.Choice("activation", ["relu", "tanh", "sigmoid"])
  lr = hp.Float("lr", min_value=0.1, max_value=0.9, step=0.4, sampling="linear")
  optimizer= hp.Choice("optimizer", ["SGD", "Adam", "Adadelta"])
  # call existing model-building code with the hyperparameter values.
  model = call_existing_code(
      units=units, optimizer_candidate=optimizer, activation=activation, lr=lr
  )
  return model

```

With hyperparameter tuning, we chose to monitor accuracy on our validation set to gauge performance of different hyperparams. From 81 trials, we found that 18 units per hidden layer, Adadelta optimization, tanh activations, and a 0.9 learning rate yields the best results, achieving an accuracy of 76.81% on the validation data. 

We plotted the model's results using a confusion matrix and followed that up with a comparison of the model's loss + accuracy on training, testing, and validation.

![Test Set Confusion Matrix](/imgs/Model2Confusion.png "Model 2 Confusion Matrix on Test Set")

To wrap up, we ensured our model's ability to  generalize by employing k-folds cross validation and plotting precision, recall, and accuracy across each fold. 

We used the following code to accomplish this:

```

estimator = KerasClassifier(model=hp_model, epochs=100    batch_size=100, verbose=0)  
kfold=RepeatedKFold(n_splits=10, n_repeats=1) 

# Perform cross validation with metrics of accuracy and MSE (cross_validate only accepts neg MSE)
scoring = ['accuracy', 'precision', 'recall']
scores = cross_validate(estimator, X_train, y_train, cv=kfold, n_jobs=1, return_train_score=True, scoring=scoring, verbose=0)

```

As you can see, we are training for 100 epochs and getting 3 metrics for each fold. Overall, we were satisfied with the results of the cross-validation, as precision and recall on the test sets hovered above 0.75 and the accuracy was close to 70%.

### Model 3

Use final_cleaned data from data preprocessing. Assign labels "Fair" and "Unfair" to each record using the method from Model 2 (if the price is greater than 1.3 * predicted price, then it's unfair)

Then, we tried to analyze top 100 features that contributes most to the classification of fair and unfair. First, we tried use pairplot as discussed in lecture. But since we have 1800+ features, it's impossible to manually analyze the relations of pairplot to select features. Therefore, suggested by ChatGPT, we use RFE.

However, feature selections was later abandoned since selecting features takes much more time to train (way more than if we just fit entire features into SVM).

We trained two SVM: linear and RBF. 

Linear is performing really well that it achieves 100% accuracy on both training and testing data.

RBF, on the other hand, was not that good. It got only about 80% initially. We applied hp tuning to it using GridSearchCV. By tuning two hp C and gamma, we successfully improved its accuracy to the same level of linear.


## Results

### First Model
Colab link to the first model: <a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model_1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Training vs test error**\
The primary measure of accuracy used for the first model was Mean Absolute Error (MAE). The training MAE ended at 242.2971, the testing MAE ended at 272.6906.\
Both of those are reasonable values for the data and the difference between them is insignificant, so there is no clear indication of either overfitting or underfitting, which is a good result because that means the model is capable of interpreting data that it has not seen before effectively. 

**Fitting graph**\
Here are some graphs of the model's predictions versus the actual data based on the number of bedrooms, bathrooms, and the total area of the apartment for each of the sets:

**Test Set**:\
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/dbb2a16d-861d-440b-8341-b925344238e1" width="500">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/e92d6a24-73f1-4639-a22e-bb4c2551af1a" width="500">
<div align = "center">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/8630b83f-9242-4ee4-ba53-1333689f5e35" width="600">
</div>

Same 3 graphs but zoomed in to avoid outliers for better display:

<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/03320355-9f0d-47ce-8596-54ee56b9334f" width="500">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/e5daccca-8566-47f7-b0bc-13dcfd8622d1" width="500">
<div align = "center">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/83a1b621-1bfd-4605-8344-a8200a9ff45d" width="600">
</div>

**Train Set**:\
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/aee82c14-9a61-4b43-a4bf-40922272a187" width="500">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/ace19504-5897-49b4-85d4-8ec0cdfd3732" width="500">
<div align = "center">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/98ed01b3-6e83-458a-b9ea-84986d019f2c" width="600">
</div>

Same 3 graphs but zoomed in to avoid outliers for better display:

<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/4835d3b4-1d65-4d0e-ab27-1d896ebedb03" width="500">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/197aa860-f3af-4123-9787-0f1292179338" width="500">
<div align = "center">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/d02c7f01-d3ad-41ae-b72f-bb25d80a27fa" width="600">
</div>

**Validation Set**:\
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/07aa396c-d0e9-489b-ac13-525a505afaff" width="500">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/a3f95083-7463-4ccd-bd17-077dee09faad" width="500">
<div align = "center">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/bd6417bb-c215-40dc-a9b9-b55b1d09c6e4" width="600">
</div>

Same 3 graphs but zoomed in to avoid outliers for better display:

<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/5ff45ece-1ba7-40c1-9dbc-d71a5a3563db" width="500">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/92f12bcf-58bb-4f81-9a3b-eb8207204056" width="500">
<div align = "center">
<img src="https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/809ab985-573c-4a64-9670-88c7db047d08" width="600">
</div>

The fitted graphs highlight the model's accuracy, particularly in matching a significant portion of the data points with their expected values, and wisely predicting average values for outliers rather than overfitting. Our analysis, especially the ReLU activation function's success, underscores the linear relationship between apartment features and rental prices. 

Although this model doesn't tackle the question of fairness directly as we do not measure or evaluate it anywhere in the model, the results demonstrate that it is possible to accurately predict what the prices should be based on the given characteristics. This foundational work paves the way for future explorations with logistic classification and SVMs, aiming for even more nuanced understandings and predictions of rental price fairness.

The training set, being the foundation of our model's learning, showed a strong correlation between features and rental prices, validating our feature selection and engineering approach. The validation set, on the other hand, served as a critical test of the model's generalizability, confirming that our model can indeed predict unseen data points with commendable accuracy. This dual success lays a solid groundwork for our future investigations into more sophisticated models and techniques, aiming to refine our predictions and insights into the dynamics of rental pricing.

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

#### Context
In attempting to predict the fairness of apartment rental prices, we developed three different models, where the first predicts rental prices, the second predicting price fairness (using logistic regression), and the third also predicting price fairness (but using SVM). 

#### For the Future
Our project is a good start in laying the foundations for further exploration in market price predicition. We can further our research by attempting to integrating other advanced machine learning techniques such as Geospatial Analysis which employs geospatial models to account the proximity of each apartment to neigbouring amenities and facilities. 

### Collaboration
Vincent Ren (Code writer & reviewer): Mainly contributed model 3, and reviewed some other code
Zhuji Zhang (Code writer): Mainly contributed on the data preprocessing, and some parts of the writeup
Sinclair Lim (Writer): Mainly contributed to README, and also contributed to direction of project, models


Our project can be found here: 
<a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/CSE151A_Group_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
