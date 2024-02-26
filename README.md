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

### First Model
Colab link to the first model: <a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/Model_1.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Training vs test error**\
The training MAE ended at 242.2971, the testing MAE ended at 272.6906.\
Both of those are reasonable values for the data and the difference between them is insignificant, so there is no clear indication of either overfitting or underfitting, which is a good result because that means the model is capable of interpreting data that it has not seen before effectively.

**Fitting graph**\
Here are some graphs of the model's predictions versus the actual data:\
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

The fitted graphs and their zoomed-in versions, although not directly viewable here, highlight the model's accuracy, particularly in matching a significant portion of the data points with their expected values, and wisely predicting average values for outliers rather than overfitting. Our analysis, especially the ReLU activation function's success, underscores the linear relationship between apartment features and rental prices. This foundational work paves the way for future explorations with logistic classification and gradient boosting machines, aiming for even more nuanced understandings and predictions of rental price fairness.

The training set, being the foundation of our model's learning, showed a strong correlation between features and rental prices, validating our feature selection and engineering approach. The validation set, on the other hand, served as a critical test of the model's generalizability, confirming that our model can indeed predict unseen data points with commendable accuracy. This dual success lays a solid groundwork for our future investigations into more sophisticated models and techniques, aiming to refine our predictions and insights into the dynamics of rental pricing.



**Next 2 models**\
Logistic Classification: For the next model, we aim to create a classification model (most likely a logistic one) that can recognize the fairness/unfairness of apartment prices, which is a more unique task.
To do this, we will add a new label to all of the entries in the table, called "fair", and then create copies of each entry with the price inflated by a random value (1.5x-2.5x), with the label "unfair".
Our logistic model will then train itself on the updated dataset.

Support Vector Machine: Like the logistic classification model, we also intend to use a support vector machine to classify prices as fair and unfair.
We will use the above-described method to create and add "unfair" entries to the table. The support vector machine will then create a separating margin between the "fair" and "unfair" classes.

**Conclusion**\
The first model used one hidden layer, with 24 units for each layer except the output and a ReLU activation function for all of them. Based on the loss function numbers described earlier in the training vs test error section, the results are reasonably accurate. This is reinforced by the graphs which demonstrate that in the vast majority of cases the predicted prices overlap with the actual prices and ignore outliers, which means that the model is accurate and was not overfitted to the data. This result makes sense because ReLU is a linear function. In the data, having a larger apartment area, more bathrooms, bedrooms, or amenities would usually correlate to a higher rent price. Therefore, a ReLU function is likely the best activation function for the data because of the near linear relationship between the variables.\
To possibly improve the model, it could be reasonable to run a more extensive keras tuner for more trials that is focused on the ReLU function specifically and only alters the number of layers, the learning rate and the number of units. We could then see if our current choice of a single hidden layer and 24 units could be tuned to be a better fit for the data.  

---
Our project can be found here: 
<a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/CSE151A_Group_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
