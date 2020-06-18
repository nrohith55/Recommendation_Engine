# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:00:18 2020

@author: Rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Recommendation_Engine\\books.csv",encoding='latin-1')

df=df.rename(columns={'Book.Title':'title','Book.Author':'author','ratings[, 3]':'rating'})


df.shape

df.columns

df.genre
#Based on Publisher
df["Publisher"].isnull().sum()# Tio find NaN values



from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english')

tfidf_matrix=tf.fit_transform(df.Publisher)


tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cos_similarity_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)

print(cos_similarity_matrix)


df_index=pd.Series(df.index,index=df['title']).drop_duplicates()

def get_title_recommendations(title,topN):

    #topN = 10
    # Getting the movie index using its title 
    df_id = df_index[title]
    
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
    df_similar_show = pd.DataFrame(columns=["title","Score"])
    df_similar_show["title"] = df.loc[df_idx,"title"]
    df_similar_show["Score"] = df_scores
    df_similar_show.reset_index(inplace=True)  
    df_similar_show.drop(["index"],axis=1,inplace=True)
    print (df_similar_show)
    #return (df_similar_show)


get_title_recommendations("Classical Mythology",topN=15)

##################################################################################################

#Based on Author
df["author"].isnull().sum()# Tio find NaN values



from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(stop_words='english')

tfidf_matrix=tf.fit_transform(df.author)


tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cos_similarity_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)

print(cos_similarity_matrix)


df_index=pd.Series(df.index,index=df['title']).drop_duplicates()

def get_title_recommendations(title,topN):

    #topN = 10
    # Getting the movie index using its title 
    df_id = df_index[title]
    
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
    df_similar_show = pd.DataFrame(columns=["title","Score"])
    df_similar_show["title"] = df.loc[df_idx,"title"]
    df_similar_show["Score"] = df_scores
    df_similar_show.reset_index(inplace=True)  
    df_similar_show.drop(["index"],axis=1,inplace=True)
    print (df_similar_show)
    #return (df_similar_show)


get_title_recommendations("Classical Mythology",topN=15)

########################################################################################





