# House Price Regression Competition
My attempt at this Kaggle competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
Using python.

Initial code structure was based on the notebook found at: https://www.kaggle.com/code/ryanholbrook/feature-engineering-for-house-prices


This code placed me at position 360 with a score of 0.12147.

## File Descriptions
| File | Description |
| --- | --- |
| data_description.txt | List of all the columns found in test.csv and train.csv, as well as all of the categories within categorical columns, and their meanings. |
| house_price_regression_code.ipynb | The code. |
| my_submission.csv | The final predictions made by the code. |
| test.csv | The data the model made its final predictions on. |
| train.csv| The data the model was trained on. |


# Structure of code
## Initial Setup and Analysis
Firstly, libraries are imported, matplotlib defaults are set, and warnings are muted.

Then, a brief look at the data is taken, categorical columns were listed and their unique categories compared to data_description.txt to highlight columns with category typos present.

Columns with NaNs were listed and counted. This allowed for some basic analysis of the data, and led to hypotheses about the nature of the missing data in the various columns.


## Loading and Processing Data
Functions to clean, encode, and finally impute data are next. These are then encapsulated in the load_data() function which as the name implies, loads in data and puts it through those aforementioned functions.


## Evaluating and Visualising Data
This processed data is then analysed, firstly Mutual Information (MI) is looked at, from this scatterplots, regplots, and swarmplots are used to visualise the top 5 features in terms of MI.

A function to score the dataset is used to establish a baseline score on the processed data, before any feature engineering takes place.


## Feature Engineering
Features are created through mathematical transforms, interactions between other features, counts of features, splitting up a single feature into multiple, and group transforms.
These features were created based on domain knowledge (for instance, the total number of bathrooms), though some of them were present in the notebook that the initial code structure was taken from already.

A heatmap was used to help find features which correlate with each other, to explore how those could be turned into new features.

Lastly a function for creating an "Outlier" column, which just highlights two houses by hand as they were both extremely large houses sold very cheaply, with saletype "partial".


## Target Encoding
CrossFoldEncoder used to reduce overfitting and avoid data leakage.


## Create Final Feature Set
The function "create_features" adds all of the newly created features to the relevant dataset and encodes them.
This new set is then scored to see whether it's improved performance.

## Analysis of New Features
The correlation between the new features and the target is viewed, regplots and swarmplots are used to visualise this correlation.
This helped reject features which weren't helpful.

## Hyperparameter Tuning and Creating The Final Submission
Hyperparameters were tuned, partially by hand and partially by HalvingRandomSearchCV for it's relatively greater speed compared to gridsearch.
I ran into overfitting issues here a few times, but performance was nonetheless raised through tuning.

Finally the model is fit, predictions are made and a csv outputted for submission.

# Further Work To Do
Currently, I haven't really explored clustering or PCA for this data. There's also much further analysis to do on most of the features, especially regarding inter-correlation.
Dropping one feature within highly correlated pairs could improve performance by reducing redundancy/multicollinearity.  Different imputation strategies also haven't been explored very much.
So far only the xgboost model has been used, it would be wise to see how other models such as lasso, elastic net, random forest, and so on, perform. From there, model stacking/blending could be explored.