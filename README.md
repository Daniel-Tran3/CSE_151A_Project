## Introduction
For our project, we are using a dataset that contains apartment rental information across the country, and it included important features such as the price, amenities, location, and number of rooms. It was chosen for the plethora of features and observations (10,000 rows) available to us, and it provided a glimpse into the housing rental market of America. Our project has three main models: the first model predicts the rental price, and we compare the predicted price with the real price to give the listing a label (fair or unfair); the second model aims to predict that label using logistic regression, and the third model aims to predict the same label using SVM. The goal for the project is to identify and predict the fairness of the price, and this is important because if our model is deployed in the real-world, it will be able to provide millions of people looking for housing crucial information on the price. With the fairness of price in hand, consumers can make a more informed decision and avoid the unfair pricing, which will eventually drive those prices down to a more reasonable level.


## Methods

### Project Objective
In this project, we will be exploring an apartment rental [dataset](https://github.com/Daniel-Tran3/CSE_151A_Project/blob/main/apartments_for_rent_classified_10K_utf.csv) and building a model to predict rental prices.

### Data Exploration

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
