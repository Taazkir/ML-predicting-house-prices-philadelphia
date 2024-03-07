# ML-predicting-house-prices-philadelphia
In this machine learning project, students will develop a model to predict the selling price of houses in Philadelphia. The project will involve various stages including data collection, preprocessing, feature engineering, model development, and evaluation.

# Data Collection

The most comprehensive data about the housing market in Philadelphia was available on the City of Philadelphia's website where I downloaded it in CSV format to use in my project. The data can be found at https://www.phila.gov/property/data/. 

The CSV file was added to the Google Colab notebook in the root directory and was imported into the project using Pandas' read_csv method.

# Data Preprocessing 

The dataset used in this project was quite comprehensive, including 82 features and over half a million data points. The first step in data preprocessing was to identify which features were relevant in determining the market value of a house. Columns that were clearly used for administrative and record-keeping purposes, such as mailing address, names of previous owners, and registry number, were dropped. Other data that were mostly missing, difficult to decipher, or not useful for residential properties were also removed. After an initial rundown of the dataset, 52 columns were eliminated, leaving 30 features. 

Next, data for properties that were not single or multi-family houses, or that did not have a market value listed, were removed. Each of the remaining features was evaluated to determine the number of classes they contained, their data type, and what these classes could mean. Attempts to drop all rows with missing values using the dropna() method proved impossible, as every row contained at least one missing value. This made handling missing values the main focus of the data preprocessing step. 

For features such as 'basements', 'central air', 'fuel', 'parcel shape', 'topography', and 'type heater', which already had a class to represent zero, unknown, other, or something along the lines of ambiguity, those missing values were clustered into the ambiguous class. For longitude and latitude, the missing values were replaced with the mean of the columns. The longitudes were recorded as negative values, so they were made positive to avoid skewing the data. For most of the features, rows with missing values were dropped.

Finally, some features were plotted using a box plot to understand what the different classes meant and how they correlated to the market value. The isna().sum() method was used to verify that the final dataset had no missing values. At the end of processing, 450 thousand lines of data remained along with 30 features. 

This project relied heavily on an article written by Nikhilesh A. Vaidya in medium.com, which was used extensively to make sense of the data.
https://medium.com/@GaussEuler/philadelphia-housing-data-part-i-data-analysis-fe45415554a9

# Feature Engineering 

Non-numeric data was encoded using binary encoding and One-Hot-Encoding. 

Here is a breakdown of the 30 features and their encoding type:

**Binary:** Basement, General Construction, Fuel, Parcel Shape, Type Heater, View type, Exterior Condition, Taxable land, Zip code, Topography 

**One Hot:** Central Air

**Numeric:** Category_code, Depth, Exempt_building, Exempt_land, Fireplaces, Frontage, Garage spaces, Homestead exemptions, Num bedrooms, Num bathrooms, Off street open, Street code, Taxable building, Total area, Total liveable area, Year build, Lat, Lng

Since there were various types of numeric values, data was standardized and scaled using SciKit learn and the interquartile range was used to find and eliminate the outliers. Lastly, Principal Component Analysis was carried out with 25, 35, 45, 55, and 65 components to reduce dimensionality and train different models to find the optimum number of features for the dataset. 

# Model Development 
I split the cleaned data into x(features) and y(market_price) variables and then trained and tested the data. I performed linear regression, KBest with linear regression, and decision trees for the PCA numerical components 25, 35, 45, 55, and 65. Providing a wide range and multiple values ensures that the models are trained on a portion of the data and will be tested on unseen data.

- Linear Regression
	The R^2 values for Linear Regression range from approximately 0.67 to close to 1, indicating moderate to high goodness of fit. However, Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) values are quite large, ranging from tens of thousands to millions. This indicates that the model's predictions deviate from the actual values by a considerable margin. Overall, Linear Regression performs reasonably well in terms of capturing the relationship between features and the target variable, but the high error metrics suggest that it might not capture all the nuances of the data.

- **SelectKBest with Linear Regression**:
	The R^2 values for KBest consistently approach 1, indicating an excellent fit to the data. The MSE and RMSE values for KBest are substantially lower compared to Linear Regression, indicating that this model's predictions are closer to the actual values. SelectKBest with Linear Regression consistently outperforms plain Linear Regression, suggesting that feature selection helps in capturing the most relevant information for predicting market_value.

- **Decision Trees**:
	The R^2 values for Decision Trees range from around 0.75 to 0.89, indicating a moderate to high goodness of fit. The MSE and RMSE values for Decision Trees are generally lower than Linear Regression but higher than KBest, suggesting that Decision Trees fall between the other two models in terms of predictive accuracy. Decision Trees provide a balance between interpretability and predictive performance, but they may not capture the complex relationships present in the data as effectively as the feature selection approach of SelectKBest.


# Discussion
The experiment shows that the model's performance improves as the number of PCA components increases. This is evidenced by the output, which shows that as the number of components increases, the R^2 value increases, and the RMSE value decreases. These results indicate better accuracy in predicting the market values of houses.

The SelectKBest model consistently achieves a perfect R^2 score with RMSE values of around 294. This shows that the model is effective in capturing the relevant features for the prediction model and reducing the number of dimensions.

The SelectKBest with linear regression performs better than normal linear regression, which suggests that the feature selection helps in capturing the most relevant information for predicting the target variable.

The model could be improved by incorporating more domain knowledge. Dealing with such a large dataset without insight into what each attribute means was challenging, and having an explanation of what the different codes stand for would have been helpful. This information would have likely improved the feature selection process. Additionally, a box plot for every feature would be useful to see the correlation between it and the market price before dropping them. Data preprocessing was a crucial step for this project and could be improved with more time. Overall, this project was a good exercise for data scientists to understand how to handle raw data and transform it into machine learning models.
