# ML-predicting-house-prices-philadelphia
In this machine learning project, students will develop a model to predict the selling price of houses in Philadelphia. The project will involve various stages including data collection, preprocessing, feature engineering, model development, and evaluation.

# Data Collection

The most comprehensive data about the housing market in Philadelphia was available on the City of Philadelphia's website where I downloaded it in CSV format to use in my project. The data can be found at https://www.phila.gov/property/data/. 

The CSV file was added to the Google Colab notebook in the root directory and was imported into the project using Pandas' read_csv method.

# Data Preprocessing 

The dataset was comprehensive and included a lot of features, 82 to be exact, and contained well above half a million data points. So, the first step of data preprocessing was to identify the features that contribute to determining the market value of a house. Some columns such as mailing address, names of previous owners, registry number, etc. which were clearly used for administrative and record-keeping purposes were dropped. Some other data that were mostly missing, difficult to decipher, or were not useful for residential properties were also dropped. After the initial rundown of the dataset, 52 columns were dropped bringing down the features used to 30. Then, I dropped data for properties that were not single or multi-family houses or that did not have a market value listed. Next, each of the features used was evaluated using the value_count() method to find the number of classes they contain, their data type, and what the classes could mean. I tried using dropna() to drop all rows with missing values since there are so many data points but it proved impossible since every row contained at least one missing value so handling missing values was the main focus of the data preprocessing step. For features such as 'basements', 'central air', 'fuel', 'parcel shape', 'topography', and 'type heater', which already had a class to represent zero, unknown, other, or something along the lines of ambiguity, those missing values were clustered into the ambiguous class. For longitude and latitude, the missing values were replaced with the mean of the columns. The longitudes were recorded as negative values so they were also made positive so as not to skew the data. Other than that for most of the features rows with missing values were dropped. Finally, some features were plotted using a box plot to understand what the different classes meant and how they correlated to the market value. The isna().sum() method was used to verify that the final dataset that will be worked on has no missing values. At the end of processing, 450k lines of data remained along with 30 features. 

This article written by Nikhilesh A. Vaidya in medium.com was extensively used to understand and make sense of the data.
https://medium.com/@GaussEuler/philadelphia-housing-data-part-i-data-analysis-fe45415554a9

# Feature Engineering 

Non-numeric data was encoded using binary encoding and One-Hot-Encoding. 

Here is a breakdown of the 30 features and their encoding type:

**Binary:** Basement, General Construction, Fuel, Parcel Shape, Type Heater, View type, Exterior Condition, Taxable land, Zip code, Topography 

**One Hot:** Central Air

**Numeric:** Category_code, Depth, Exempt_building, Exempt_land, Fireplaces, Frontage, Garage spaces, Homestead exepmtion, Num bedrooms, Num bathrooms, Off street open, Street code, Taxable building, Total area, Total liveable area, Year build, Lat, Lng

Since there were various types of numeric values, data was standardized and scaled using SciKit learn and the interquartile range was used to find and eliminate the outliers. Lastly, Principal Component Analysis was carried out with 25, 35, 45, 55, and 65 components to reduce dimensionality and train different models to find the optimum number of features for the dataset. 

