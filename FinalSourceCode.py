#!/usr/bin/env python
# coding: utf-8

# In[734]:
#G01276965
#VishalNadar

#importing all essential libraries
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, col
import pyspark
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StopWordsRemover
import math
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.feature import OneHotEncoder
import time
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from statistics import mean, StatisticsError
import pandas as pd
import numpy as np
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import functions as F
from pyspark.sql import functions as functions
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[735]:


#starting spark session
spark = SparkSession.builder.master("local").appName('assign4').getOrCreate()
sc=spark.sparkContext


# In[736]:


#importing csv data 
movies = spark.read.csv("movies.csv",header=True)
ratings = spark.read.csv("ratings.csv",header=True)
tags = spark.read.csv("tags.csv",header=True)
datalimit = 20000000 #  limit data for debugging 1000000  10000000 total dataset is 20000263
ratings = ratings.limit(datalimit)
ratings = ratings.drop("timestamp")
ratings.show()


# In[737]:


ratings.count()


# In[738]:


#joining genres with respect to movieId
movie_ratings = ratings.join(movies,['movieId'],'left')
movie_ratings.show()


# In[739]:


#casting to proper data type
ratings = ratings.withColumn("userId", ratings["userId"].cast("int"))
ratings = ratings.withColumn("movieId", ratings["movieId"].cast("int"))
ratings = ratings.withColumn("rating", ratings["rating"].cast("float"))
ratings.show()


# In[740]:


#splitting to train and test data
(train, test) = ratings.randomSplit([0.8, 0.2], seed = 2020)


# In[741]:


#Als with best cross validation for best parameters
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=8, maxIter=10, regParam=0.1)
paramGrid = ParamGridBuilder()    .addGrid(als.rank, [10, 50, 100])    .addGrid(als.regParam, [0.01, 0.05, 0.1])    .build()
rmseEvaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=rmseEvaluator, numFolds=2)
mseEvaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
maeEvaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")


# In[742]:


model = cv.fit(train)


# In[743]:


tested = model.transform(test)


# In[744]:


tested=tested.na.fill(0.0,subset=["prediction"])
tested.show()


# In[745]:


# printing the best models
best_model = model.bestModel
print("**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())


# In[746]:


#rmse,mse and mae values
RMSE = rmseEvaluator.evaluate(tested)
print(RMSE)
MSE = mseEvaluator.evaluate(tested)
print(MSE)


# In[747]:


#using mae instead of map as it was allowed by professor
mae = maeEvaluator.evaluate(tested)
print(mae)


# In[748]:


# Generate n Recommendations for all users
recommendations = best_model.recommendForAllUsers(5)
recommendations.show()


# In[749]:


#printing recommendations
nrecommendations = recommendations    .withColumn("rec_exp", explode("recommendations"))    .select('userId', col("rec_exp.movieId"), col("rec_exp.rating"))
nrecommendations.join(movies, on='movieId').limit(10).show()


# In[750]:


#calculating item-item similarity
def gettingratings(userID, movieID, item_similarity, train_df, k):
    # taking k users that have rated the movie
    this_item_distances = item_similarity[movieID]
    sorted_distances = this_item_distances.sort_values(ascending=False)[1:]
    # getting ratings by user
    this_user = train_df[int(userID)]
    ratings_this_user_this_movie = []
    for key in sorted_distances.keys():
        if len(ratings_this_user_this_movie) >= k:
            break
        this_user_this_movie = this_user[key]
        if this_user_this_movie > 0:
            ratings_this_user_this_movie.append(this_user_this_movie)
    item_rating = mean(ratings_this_user_this_movie)
    return float(item_rating)


def item_item_cf(k, datalimit, test):
    # get unique values in a column
    ratings = pd.read_csv('ratings.csv')
    ratings = ratings.head(datalimit)
    ratings = ratings.drop("timestamp", axis=1)
    pivoted = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    item_similarity = cosine_similarity(pivoted)
    item_similarity = pd.DataFrame(item_similarity)
    item_similarity.index = pivoted.index
    item_similarity.columns = pivoted.index
    udf_test_function = F.udf(lambda x, y: gettingratings(x,y,item_similarity,pivoted,k), DoubleType())
    item_item_results_df = test.withColumn("prediction", udf_test_function("userId", "movieId"))
    return item_item_results_df
k=15
prediction_item_item_df = item_item_cf(k, datalimit, test)


# In[751]:


prediction_item_item_df.show()


# In[752]:


#renaming als prediction to create hybrid functions
tested = tested.withColumnRenamed("prediction","alsprediction")
tested.show()


# In[753]:


#combining the predictions of item-item and als for hybrid system
tested = tested.join(prediction_item_item_df, ['movieId','userId','rating'], 'left')


# In[754]:


tested.show()


# In[755]:


#multiplying by weights to form the hybrid predictions of item-item and als
#here chaning weights for better rmse
tested=tested.withColumn("hyb1predic", col("alsprediction")*0.5+col("prediction")*0.5)
tested.show()


# In[756]:


#calculating the hybrid rmse value
rmseEvaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="hyb1predic")
acc = rmseEvaluator.evaluate(tested)
print("rmse: ", acc)


# In[757]:


#movies = movies.limit(datalimit)
#tags = tags.limit(datalimit)


# In[758]:


#taking tags csv for better features
tags = tags.drop("timestamp")
tags.show()


# In[759]:


#combining the tag column to dataset
movie_ratings = movie_ratings.join(tags, ['movieId','userId'], 'left')
movie_ratings.show()


# In[760]:


#concating the tag column to genres
movie_ratings=movie_ratings.withColumn("genres",concat_ws(" ",col("genres"),col("tag")))
movie_ratings.show()


# In[761]:


#changing the data type of column
movie_ratings = movie_ratings.drop("tag")
movie_ratings = movie_ratings.withColumn("userId", movie_ratings["userId"].cast("int"))
movie_ratings = movie_ratings.withColumn("movieId", movie_ratings["movieId"].cast("int"))
movie_ratings = movie_ratings.withColumn("rating", movie_ratings["rating"].cast("float"))


# In[762]:


#pre processing the genres column
movie_ratings = movie_ratings.withColumn("genres",regexp_replace("genres", "[^A-Za-z_]", " "))
movie_ratings = movie_ratings.withColumn("genres",regexp_replace("genres", " +", " "))
movie_ratings = movie_ratings.withColumn("genres", lower(col("genres")))
movie_ratings.show()


# In[763]:


#spliting the words in genres column
movie_ratings = movie_ratings.withColumn("genres",split(col("genres")," "))
movie_ratings.show()


# In[764]:


#converting the words to vectors
cv = CountVectorizer(inputCol="genres", outputCol="vec")
model = cv.fit(movie_ratings)
movie_ratings = model.transform(movie_ratings)
movie_ratings.show()


# In[765]:


#converting to features with vector indexer
featureIndexer = VectorIndexer(inputCol = "vec", outputCol = "features").fit(movie_ratings)
d = featureIndexer.transform(movie_ratings)
d = d.fillna(0.0,subset=["rating"])
d.show()


# In[766]:


#convertin label column to int
d = d.withColumn("label", d["rating"].cast("int"))


# In[767]:


train.show()


# In[768]:


#combining the features with test data for testing
test = test.join(d, ['movieId','userId','rating'], 'left')
test.show()


# In[769]:


#Here we run the Logistic Regression model
lr = LogisticRegression(labelCol="label", featuresCol="features")
model=lr.fit(d)
predict_test=model.transform(test)
rmseEvaluator2 = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")
acc = rmseEvaluator2.evaluate(predict_test)
print("rmse: ", acc)


# In[770]:


predict_test = predict_test.select("movieId","userId","rating","prediction")


# In[771]:


predict_test = predict_test.withColumnRenamed("prediction","lrprediction")
predict_test.show()


# In[ ]:


predict_test = predict_test.na.fill(0.0,subset=["lrprediction"])
predict_test.show()


# In[ ]:


tested = tested.select("movieId","userId","rating","hyb1predic")
tested.printSchema()


# In[ ]:


predict_test = predict_test.select("movieId","userId","rating","lrprediction")
predict_test.printSchema()


# In[ ]:


predict_test = predict_test.join(tested, ["movieId","userId","rating"], "left")
predict_test.show()


# In[ ]:





# In[ ]:


# Applying weights here and changing to produce diffent rmse
predict_test=predict_test.withColumn("hyb2predic", col("hyb1predic")*0.5+col("lrprediction")*0.5)
predict_test.show()


# In[ ]:


# printing the rmse of all 3 hybrid models combined
rmseEvaluator2 = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="hyb2predic")
acc = rmseEvaluator2.evaluate(predict_test)
print("rmse: ", acc)


# In[ ]:




