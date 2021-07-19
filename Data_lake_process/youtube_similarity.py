from google_spreadsheet_api.function import get_df_from_speadsheet, get_gsheet_name, update_value
from youtube_dl_fuction.fuctions import get_youtube_title_and_youtube_uploader_from_youtube_url
from support_function.text_similarity.text_similarity import get_token_set_ratio
import time
import pandas as pd
from core.models.data_source_format_master import DataSourceFormatMaster
from google_spreadsheet_api.create_new_sheet_and_update_data_from_df import creat_new_sheet_and_update_data_from_df
import numpy as np


def similarity(track_title: str, youtube_url: str, formatid: str, duration: str = None):
    special_characters = \
        get_df_from_speadsheet(gsheet_id='1W1TlNDXqZTMAaAFofrorqaEo6bfX7GjwnhWMXcq70xA', sheet_name='Similarity')[
            'Keywords'].tolist()
    track_title = track_title.lower()
    get_youtube_info = get_youtube_title_and_youtube_uploader_from_youtube_url(youtube_url)
    get_youtube_title = get_youtube_info['youtube_title'].lower()
    # get_youtube_uploader = get_youtube_info['uploader'].lower()
    get_youtube_duration = get_youtube_info['duration']
    abs_duration = abs(int(duration) - int(get_youtube_duration))

    if abs_duration > 10000 and formatid == DataSourceFormatMaster.FORMAT_ID_MP3_FULL:
        token_set_ratio = 0
    else:
        result = "type 3"
        for special_character in special_characters:
            if special_character in track_title:
                result = "type 1"
                break
            elif special_character in get_youtube_title:
                result = "type 2"
                break
            else:
                pass
        if result == "type 1":
            if special_character in get_youtube_title:
                token_set_ratio = get_token_set_ratio(get_youtube_title, track_title)
            else:
                token_set_ratio = 0
        elif result == "type 2":
            if special_character in track_title:
                token_set_ratio = get_token_set_ratio(get_youtube_title, track_title)
            else:
                token_set_ratio = 0
        else:
            token_set_ratio = get_token_set_ratio(get_youtube_title, track_title)

    get_youtube_info['similarity'] = token_set_ratio
    youtube_info = get_youtube_info
    return youtube_info


if __name__ == "__main__":
    # https://docs.google.com/spreadsheets/d/1aRhZ7NQAfhud3jjR5aboCZ3Ew8u2Y0SqGqUQYwcUnBs/edit#gid=98817891
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 50, 'display.width', 1000)
    # https://docs.google.com/spreadsheets/d/1eO8J2qqjxgRVnc3b1EWGskVHYc1baUAmDzdqT6hIdRg/edit#gid=926860952
    gsheet_id = '1bKbrX9lul1njeUgBHgPvuzpmuCes7XNVP6AAlIqXONw'
    sheet_name = 'mp_4'
    df = get_df_from_speadsheet(gsheet_id=gsheet_id, sheet_name=sheet_name)
    df["DurationMs"].replace({"": "0"}, inplace=True)
    df = df.loc[:]
    row_index = df.index
    start = row_index.n_estimators_start
    stop = row_index.n_estimators_stop
    step = 25
    for i in range(start, stop, step):
        x = i + step
        if x <= stop:
            stop_range = x
        else:
            stop_range = stop
        f = []
        for j in range(i, stop_range):
            track_title = df.track_title.loc[j]
            SourceURI = df.SourceURI.loc[j]
            FormatID = df.FormatID.loc[j]
            DurationMs = df.DurationMs.loc[j]
            k = similarity(track_title=track_title, youtube_url=SourceURI, formatid=FormatID, duration=DurationMs).get('similarity')
            f.append([k])
        joy1 = f"{sheet_name}!N{i+2}"
        update_value(list_result=f, grid_range_to_update=joy1, gsheet_id=gsheet_id)

    print("--- %s seconds ---" % (time.time() - start_time))
