import youtube_dl
import time
from youtube_dl.utils import DownloadError
from core.crud.sql.datasource import get_one_youtube_url_and_youtube_uploader_by_youtube_url
from core import youtube_com_cookies_path
import json
from numpy import random
import traceback


def get_raw_youtube_info(youtube_url: str):
    ytdl_options = {
        # "cachedir": False,
        "quiet": True,
        "nocheckcertificate": True,
        "restrictfilenames": True,
        "cookiefile": youtube_com_cookies_path,
    }

    ydl = youtube_dl.YoutubeDL(ytdl_options)

    result = ydl.extract_info(
        youtube_url,
        # download=False  # We just want to extract the info
        download = True  # We just want to extract the info
    )
    joy = json.dumps(result)
    print(joy)
    return joy


def get_raw_title_uploader_from_youtube_url(youtube_url: str):
    ytdl_options = {
        # "cachedir": False,
        "quiet": True,
        "nocheckcertificate": True,
        "restrictfilenames": True,
        "cookiefile": youtube_com_cookies_path,
    }

    try:
        ydl = youtube_dl.YoutubeDL(ytdl_options)

        result = ydl.extract_info(
            youtube_url,
            download=False  # We just want to extract the info
        )
        youtube_info_result = {"youtube_url": youtube_url, "uploader": result.get('uploader'),
                               "youtube_title": result.get('title'), "duration": result.get('duration') * 1000}
    except DownloadError as ex:
        youtube_info_result = {"youtube_url": youtube_url, "uploader": f"{ex}", "youtube_title": f"{ex}", "duration": f"0"}
    except:  # noqa
        youtube_info_result = {"youtube_url": youtube_url, "uploader": "Error: Unknown error",
                               "youtube_title": "Error: Unknown error", "duration": "0"}
    x = random.uniform(0.5, 3)
    time.sleep(x)
    # print(youtube_info_result)
    return youtube_info_result


def get_youtube_title_and_youtube_uploader_from_youtube_url(youtube_url: str):
    db_datasources = get_one_youtube_url_and_youtube_uploader_by_youtube_url(youtube_url)

    if not db_datasources:
        youtube_info_result = get_raw_title_uploader_from_youtube_url(youtube_url)
    else:

        for db_datasource in db_datasources:
            info = db_datasource.info.get('source', None)
            youtube_title = info.get('title', None)
            uploader = info.get('uploader', None)
            duration = db_datasource.duration_ms
            youtube_info_result = {"youtube_url": youtube_url, "uploader": uploader,
                                   "youtube_title": youtube_title, "duration": duration}
    print(youtube_info_result)
    return youtube_info_result


if __name__ == "__main__":
    start_time = time.time()
    youtube_urls = [
        "https://www.youtube.com/watch?v=S5sglWkx064",
        "https://www.youtube.com/watch?v=K4ASDUU9CcA"
    ]
    for youtube_url in youtube_urls:
        print(youtube_url)
        get_raw_youtube_info(youtube_url=youtube_url)
        # get_youtube_title_and_youtube_uploader_from_youtube_url(youtube_url)
        # joy = get_raw_title_uploader_from_youtube_url(youtube_url)
        # print(joy)
    t2 = time.time() - start_time
    print(t2)

