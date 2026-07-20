import json
import yt_dlp
import subprocess
import glob
import os

ydl_opts = {
    "skip_download" : True,
    "write_comments" : True,
    "quiet":True,
    "verbose":True,
    "outtmpl": "%(id)s.%(ext)s",
    "js_runtimes": {"node": {}}
}

def read_json(json_path)->dict:
    "Read configuration file. Return a json."
    with open(json_path) as f:
        return json.load(f)

def get_url(value)->list:
    "Return a list of url."
    return value["url"]

def get_director(value)->str:
    "Return a string with the director's name."

    return value["director"]
   
def get_video_data(url)->list[dict]:
    "Return a json with video data."
    result = subprocess.run(
        [
            "yt-dlp",
            "--dump-json",
            "--write-comments",
            "--skip-download",
            url
        ],
        capture_output=True,
        text=True
    )

    return json.loads(result.stdout)

def build_comment_threads_from_data(data)->list[dict]:
    """ 
    Returns:
        list[dict]: list of dictionaires containing information about each video. The columns are:
        id of the comment
        author
        text
        likes
        timestamp
        if the comment is replied
        parent comment
        parent author
        parent text
    """
    comments = data.get("comments", [])
    if not comments:
        return []

    by_id = {c["id"]: c for c in comments}
    result = []

    for c in comments:
        parent_id = c.get("parent")

        if parent_id == "root":
            parent_text = None
            parent_author = None
        else:
            parent = by_id.get(parent_id)
            parent_text = parent.get("text") if parent else None
            parent_author = parent.get("author") if parent else None

        result.append({
            "id": c.get("id"),
            "author": c.get("author"),
            "text": c.get("text"),
            "likes": c.get("like_count", 0),
            "timestamp": c.get("timestamp"),
            "is_reply": parent_id != "root",
            "parent_id": parent_id if parent_id != "root" else None,
            "parent_author": parent_author,
            "parent_text": parent_text,
        })

    return result

def process_videos(list_urls, movie, director)->list[dict]:
    "Returns a list of dictionaries, assigning each comment (and its metadata) to the respective url and movie."
    all_videos = []

    for url in list_urls:
        print(f"Processing: {url}")

        try:
            data = get_video_data(url)

            video_entry = {
                "movie":movie,
                "director":director,
                "video_id": data.get("id"),
                "title": data.get("title"),
                "url": data.get("webpage_url"),
                "comments": build_comment_threads_from_data(data)
            }

            all_videos.append(video_entry)

            print(f"✔ {data.get('title')} — {len(video_entry['comments'])} commenti")

        except Exception as e:
            print(f"❌ Errore con {url}: {e}")

    return all_videos


def save_output(all_videos, folder_path="Data", output_file="comments_with_context.json")->list[dict]:  
    "Return a json with all info about comments and video."  
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, output_file)
    with open(full_path, "w", encoding="utf-8") as f:  # ← "w" non "a"
        json.dump(all_videos, f, ensure_ascii=False, indent=2)
    print(f"\nSalvato in {full_path}")

def run_pipeline()->None:
    """
    For each movie in the configuration json file:
        1. Extract the movies'url and director from the config file;
        2. Extract reviews from each url
        3. Create a json file with:
                movie
                director
                video_id
                title
                url
                comments
                likes
                timestamp
                is_reply
                parent_id
                parent_author
                parent_text
        4. Save the json file
    """
    all_videos = []
    file_json = read_json(r"Data\Config.json")
    for movie,value in file_json.items():
        list_urls = get_url(value= value)
        director = get_director(value= value)
        videos = process_videos(list_urls=list_urls, movie= movie, director= director)
        all_videos.extend(videos)
    save_output(all_videos)


if __name__ == "__main__":
    run_pipeline()