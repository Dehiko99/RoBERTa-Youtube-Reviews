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

connection_string = ("YOUR_CONNECTION_STRING")

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

def get_actor_to_character(movie,config_json):
    cast_dict = config_json[movie]["cast"]
    return {v: k for k, v in cast_dict.items()}

def get_cast(movie,config_json):
    "Return movie's cast."
    cast_dict = config_json[movie]["cast"]
    return cast_dict.keys()


def replace_actors_with_comments(text):
    for actor, character in actor_to_character.items():
        # replace exact matches with comment
        text = re.sub(rf"\b{re.escape(actor)}\b", f"# {character}", text)
    return text

def create_zeroshot_topic_list():
    "Return a list with the topics to be predict."
    return [
        # film's attributes
        "acting and cast performance",
        "plot and story",
        "film duration and pacing",
        "director and directorial choices", 
        "cinematography and visual style",
        "visual effects and graphics",
        "soundtrack and music",
        "characters and character development",
        # comparison
        "comparison with previous films in the saga",
        "comparison with other films",
        # general opinion
        "overall opinion and recommendation",
        "emotional reaction to the film",
        # out of topic
        "spam and advertisement",
        "off topic",
    ]

def clean_dataset(df):
    "Clean text column."
    df = df.dropna(subset=["text"])
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip() != ""]
    return df

def return_docs(df):
    "Return a list of comments."
    return df["text"].astype(str).tolist()

def pipeline_topic_classifier():
    "Return a pipeline to predict the topic."
    return pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=0 #-1 if cpu is not available
    )

def return_director_name(config_json,movie):
    "Return director's name of the movie."
    return config_json[movie]["director"]

def assign_topic(df, topic_classifier, director_name, cast, threshold=0.5, batch_size=32):
    """
    Predict topics. Return a datafame with 
    Comment
    Primary Topic
    Primary Score
    All topics
    All scores 
    """
    all_results = []
    
    director_names = [n.lower() for n in director_name.split()]
    cast_names = [n.lower() for n in cast]

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        
        results = topic_classifier(
            batch,                   
            candidate_labels=zeroshot_topic_list,
            multi_label=True
        )
        
        for doc, result in zip(batch, results):
            topic_scores = [(result['labels'][i], result['scores'][i]) 
                            for i in range(len(result['labels']))]
            relevant = [(l, s) for l, s in topic_scores if s >= threshold]
            
            if not relevant:
                relevant = [("off topic", 1.0)]
            
            # post-processing: correct only if the director's name is in the comment
            doc_lower = doc.lower()
            if any(name in doc_lower for name in director_names):
                relevant = [(l, s) for l, s in relevant 
                            if l != "acting and cast performance"]
                relevant.insert(0, ("director and directorial choices", 1.0))
                
            if any(name in doc_lower for name in cast_names):
                relevant = [(l, s) for l, s in relevant 
                            if l != "director and directorial choices"
                            and l != "characters and character development"]
                relevant.insert(0, ("acting and cast performance", 1.0))
            
            all_results.append({
                "comment": doc,
                "primary_topic": relevant[0][0],
                "primary_score": relevant[0][1],
                "all_topics": [l for l, s in relevant],
                "all_scores": [s for l, s in relevant],
            })
        
        print(f"Processed {min(i+batch_size, len(docs))}/{len(docs)} comments")
    
    return pd.DataFrame(all_results)

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

def get_offtopic_comments(df_topic):
    "Return a list of comments predicted as out of topic."
    return df_topic[df_topic["primary_topic"].isin(["off topic","off_topic"])]["comment"].unique()

def remove_offtopic_comments(df_topic, offtopic_comments):
    "Remove comments predicted as out of topic"
    return df_topic[~df_topic["comment"].isin(offtopic_comments)]

def get_df_exploded(df):
    "Explode df by all topics and return a dataframe with scores above the threshold"
    df = df.explode(["all_topics","all_scores"])
    df[['all_topics', "all_scores"]] = df[['all_topics', 'all_scores']].applymap(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    return df


json_comments = load_comments_json()
json_config = load_config_json()
df = create_dataset(json_comments= json_comments)
df_movies = pd.read_sql("SELECT * FROM Movie", con=engine)

zeroshot_topic_list = create_zeroshot_topic_list()
topic_classifier = pipeline_topic_classifier()
sentiment_pipeline = return_sentiment_pipeline()

for movie in df["movie"].unique():
    director_name = return_director_name(config_json= json_config, movie= movie) 
    actor_to_character = get_actor_to_character(movie, config_json= json_config)
    cast = get_cast(movie, config_json=json_config)
    df_filtered = df[df["movie"]==movie]
    df_filtered["text"] = df_filtered["text"].apply(replace_actors_with_comments)
    df_filtered = clean_dataset(df= df_filtered)
    docs = return_docs(df = df_filtered)
    ### Sentiment Classifier
    df_filtered[["sentiment_label","sentiment_score"]] = df_filtered["text"].apply(lambda x: pd.Series(get_sentiment(x)))
    df_filtered = pd.merge(df_filtered, df_movies, left_on=["movie"], right_on=["Movie"]).drop("movie", axis = 1)
    ### Topic Classifier
    df_topic_classified = assign_topic(df_filtered,topic_classifier=topic_classifier,cast= cast, director_name=director_name)
    df_topic_classified["ID Comment"] = df_filtered["id"] #insert id to identify comments
    df_topic_classified['ID Movie'] = df_filtered['ID Movie'] #insert id movie
    offtopic_comments = get_offtopic_comments(df_topic=df_topic_classified)
    df_topic_classified_filtered = remove_offtopic_comments(df_topic= df_topic_classified, offtopic_comments= offtopic_comments)#removing reviews predicted out of topic as primary topic
    df_topic_classified_filtered_exploded = get_df_exploded(df= df_topic_classified_filtered)
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

    df_topic_classified_filtered_exploded = df_topic_classified_filtered_exploded[['ID Movie', 'ID Comment', 'primary_topic', 'primary_score', 'all_topics', 'all_scores']].rename(columns={
    'primary_topic': 'Primary Topic',
    'primary_score': 'Primary Score',
    'all_topics': 'Topic',
    'all_scores': 'Topic Score'
    })

    df_topic_classified_filtered_exploded.to_sql(
            name="Topic Forecast",
            con=engine,
            if_exists="append",
            index=False
        )
