import pandas as pd
import json
import numpy as np
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import warnings
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.exc import SAWarning

warnings.filterwarnings("ignore", category=SAWarning)

server = "SERVER_NAME" 
database = "Movies"

connection_string = (
   "YOUR_CONNECTION_STRING"
)

engine = create_engine(connection_string)

def load_comments_json():
    "Return a json with comments."
    with open(r"Data\comments_with_context.json", encoding="utf8") as f:
        return json.load(f)

def load_config_json():
    "Return json with configurations."
    with open(r"Data\config.json", encoding="utf8") as f:
        return json.load(f)

def create_dataset(json_comments):
    comments = []
    for video in json_comments:  
        for item in video["comments"]:
            comments.append({
                ""
                "movie": video.get("movie"),
                "director":video.get("director"),
                "video_id":      video.get("video_id"),   
                "video_title":   video.get("title"),
                "id":            item.get("id"),
                "author":        item.get("author"),
                "text":          item.get("text"),
                "likes":         item.get("likes"),
                "timestamp":     item.get("timestamp"),
                "is_reply":      item.get("is_reply"),
                "parent_id":     item.get("parent_id"),
                "parent_author": item.get("parent_author"),
                "parent_text":   item.get("parent_text"),
            })
    return pd.DataFrame(comments)


##################################################################

###################################################################

def clean_dataset(df):
    "Clean text column."
    df = df.dropna(subset=["text"])
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]
    return df

def return_docs(df):
    "Return a list of comments."
    return df["text"].astype(str).tolist()

def return_sentiment_pipeline():
    "Return a pipeline to predict the sentiment."
    return pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    model_kwargs={"use_safetensors": True}
    )

def get_sentiment(text):
    "Return sentiment label and score."
    result = sentiment_pipeline(text[:512])[0]
    return result["label"],result["score"]

if __name__ == "main":
    json_comments = load_comments_json()
    json_config = load_config_json()
    df = create_dataset(json_comments= json_comments)
    df_movies = pd.read_sql("SELECT * FROM Movie", con=engine)
    sentiment_pipeline = return_sentiment_pipeline()

    for movie in df["movie"].unique():
        df_filtered = df[df["movie"]==movie]
        df_filtered = clean_dataset(df= df_filtered)
        docs = return_docs(df = df_filtered)
        ### Sentiment Classifier
        df_filtered[["sentiment_label","sentiment_score"]] = df_filtered["text"].apply(lambda x: pd.Series(get_sentiment(x)))
        df_filtered = pd.merge(df_filtered, df_movies, left_on=["movie"], right_on=["Movie"]).drop("movie", axis = 1)
        df_filtered = df_filtered[['ID Movie',"video_id","id","sentiment_label","sentiment_score"]].rename(columns={
        "video_id":"ID Video",
        "id":"ID Comment",
        "sentiment_label":"Sentiment Forecast",
        "sentiment_score":"Sentiment Score"
        }) 

        df_filtered.to_sql(
            name="Sentiment Forecast",
            con=engine,
            if_exists="append",
            index=False
        )
