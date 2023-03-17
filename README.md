# Recommender-System
Introduction: -

The goal of the assignment is to use recommender system to predict ratings of movies. We are using the dataset from 20M MovieLens dataset. We will be using different models of recommender systems along with hybrid systems to predict the movie ratings.

Explanation: -

• Imported necessary spark libraries and spark session. Imported csv file with spark.read.csv.
• Using join function to merge genres data into the ratings dataset.
• Changed the ratings data types to proper format.
• Spitted the dataset to 80:20 ratio as train and test.
• Used ALS algorithm with grid and CrossValidator to get the best parameters for ALS.
• With the prediction, printed the best model and evaluated RMSE, MSE AND MAE values.
• Printed the recommendations we get from the ALS model just to see how it works to suggest movies.
• Creating an item-item similarity function which calculates the cosine similarity to predict the ratings of each movie.
• With the item-item rating data frame I applied different weights to ALS ratings and item-item ratings.
• Combined and stored the new predictions under hyb1predic.
• Importing tags and genres data into data frame.
• Combining and preprocessing the tags and genres and then using CountVector and VectorIndexer to create the features to use with linear regression.
• Joined the LR model to fit the test data to include its features.
• Taking the predictions and again applying the weights to LRprediction and hyb1predic to create a hybrid prediction for 3 systems.
• Applying RMSE to find the performance of different weights of each of the three systems.
• For running the spark program. I used a py version of ipynb program with the help of AWS S3, AWS EMR and putty to run the code on a cluster.

Conclusion: -

I have learned to create Hybrid recommendations systems which suggests movies to user and predicts the movie ratings with respect to the user. Created an item-item CF which predicts ratings based on the movies seen by similar users. The model predicts better rating with more
and more hybrid models. Adjusting the weights with respect to the RMSE of individual models gives better predictions and the three-model hybrid system gives the best RMSE model.

Instructions to run code:
1) install proper spark version and pip install all necessary libraries.
2) open the code stored in code/ FinalSourceCode.py.
3) run the program.
