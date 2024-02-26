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
![image](https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/dbb2a16d-861d-440b-8341-b925344238e1)
![image](https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/e92d6a24-73f1-4639-a22e-bb4c2551af1a)
![image](https://github.com/Daniel-Tran3/CSE_151A_Project/assets/44418360/8630b83f-9242-4ee4-ba53-1333689f5e35)

Given how much the points overlap, that is a good indicator that the model is accurate since a lot of the datapoints match their expectations. Furthermore, there are some outliers in the actual data that the model did not predict correctly, instead predicting a more average value, which is once again a good sign in the sense that the model did not overfit on the data.

**Next 2 models**\
Linear Regression: **TODO**

Gradient boosting machine: **TODO**

**Conclusion**\
The first model used one hidden layer, with 24 units for each layer except the output and a ReLU activation function for all of them. Based on the loss function numbers described earlier in the training vs test error section, the results are reasonably accurate. This is reinforced by the graphs which demonstrate that in the vast majority of cases the predicted prices overlap with the actual prices and ignore outliers, which means that the model is accurate and was not overfitted to the data. This result makes sense because ReLU is a linear function. In the data, having a larger apartment area, more bathrooms, bedrooms, or amenities would usually correlate to a higher rent price. Therefore, a ReLU function is likely the best activation function for the data because of the near linear relationship between the variables.\
To possibly improve the model, it could be reasonable to run a more extensive keras tuner for more trials that is focused on the ReLU function specifically and only alters the number of layers, the learning rate and the number of units. We could then see if our current choice of a single hidden layer and 24 units could be tuned to be a better fit for the data.  

---
Our project can be found here: 
<a target="_blank" href="https://colab.research.google.com/github/Daniel-Tran3/CSE_151A_Project/blob/main/CSE151A_Group_Project.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
