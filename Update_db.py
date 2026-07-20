import json
import pandas as pd
from sqlalchemy import create_engine
import warnings
from sqlalchemy.exc import SAWarning
import os
warnings.filterwarnings("ignore", category=SAWarning)

server = "SERVER_NAME"   # or server name
database = "Movies"

connection_string = (
    "YOUR_CONECTION_STRING"
)


engine = create_engine(connection_string)

def get_json_config(path = r"Data\config.json"):
    with open(path,"r") as f:
        return json.load(f)

def get_movie(file):
    list_movies = []
    for movie in file.keys():
        list_movies.append({"Movie":movie})
    return pd.DataFrame(list_movies)

def get_urls(file):
    rows_urls= []
    for movie,values in file.items():
        list_url = values["url"] 
        director = values["director"]

        for index, url in enumerate(list_url, start=1):
            rows_urls.append({
                "Movie":movie,
                "ID url": index,
                "Url":url,
            })
    return pd.DataFrame(rows_urls)

def get_director(file):
    rows_director = []
    for movie,values in file.items():
        director = values["director"]
        rows_director.append({"Director":director})

    return pd.DataFrame(rows_director)

def write_to_db(name, df):
    df.to_sql(
        name=name,
        con=engine,
        if_exists="append",
        index=False
    )

def get_json_comments(path = r"Data\comments_with_context.json"):
    comments = []
    with open(path,"r") as f:
        data =  json.load(f)
        for video in data:  # <-- livello video mancava
            for item in video["comments"]:
                comments.append({
                    "ID Video":      video.get("video_id"),   # utile per tracciare la fonte
                    "Video Title":   video.get("title"),
                    "ID Comment":            item.get("id"),
                    "Text":          item.get("text"),
                    "Likes":         item.get("likes")
                })
    return pd.DataFrame(comments)

if __name__ == "__main__":
    engine = create_engine(connection_string)
    config_json = get_json_config()
    df = get_movie(file=config_json)
    write_to_db(name="Movie", df=df)
    df_movies = pd.read_sql("SELECT * FROM Movie", con=engine)
    df_urls = get_urls(file=config_json)
    write_to_db(name="Url", df=df_urls)
    df_urls = pd.merge(df_urls, df_movies, on=["Movie"]).drop("Movie", axis=1)
    df_director = get_director(file=config_json)
    write_to_db(name="Director", df=df_director)
    df_comments = get_json_comments()
    df_videos = df_comments[["ID Video","Video Title"]].drop_duplicates()
    write_to_db(name="Video",df=df_comments)
    df_yt_comments = df_comments[['ID Video', 'ID Comment', 'Text', 'Likes']]
    write_to_db(name="Youtube Comments",df=df_yt_comments)
