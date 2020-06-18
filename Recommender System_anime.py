# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:24:30 2020

@author: Rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Recommendation_Engine\\df.csv")

df.shape

df.columns

df.genre

df["genre"].isnull().sum()# Tio find NaN values

df['genre']=df['genre'].fillna(" ") # Replacimg the null values  with  empty strings

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english')

tfidf_matrix=tf.fit_transform(df.genre)


tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cos_similarity_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)

print(cos_similarity_matrix)


df_index=pd.Series(df.index,index=df['name']).drop_duplicates()

def get_anime_recommendations(Name,topN):

    #topN = 10
    # Getting the movie index using its title 
    df_id = df_index[Name]
    
    # Getting the pair wise similarity score for all the df's with that 
    # df
    cosine_scores = list(enumerate(cos_similarity_matrix[df_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar df's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the df index 
    df_idx  =  [i[0] for i in cosine_scores_10]
    df_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    df_similar_show = pd.DataFrame(columns=["name","Score"])
    df_similar_show["name"] = df.loc[df_idx,"name"]
    df_similar_show["Score"] = df_scores
    df_similar_show.reset_index(inplace=True)  
    df_similar_show.drop(["index"],axis=1,inplace=True)
    print (df_similar_show)
    #return (df_similar_show)


get_anime_recommendations("Ginga Eiyuu Densetsu",topN=15)




















