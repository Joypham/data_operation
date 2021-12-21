import time

if __name__ == "__main__":
    start_time = time.time()

    # Artist_page check_box: comment
    # from tools.artistpage_justin import check_box
    # check_box()

    # Extract artist_name for collecting from youtube: comment
    from tools.daily_export_artist_contribution import export_artist_contribution
    export_artist_contribution()

    # Extract wiki, lyric single_page, album_page: comment
    from tools.daily_singlepage_albumpage import upload_album_wiki, upload_track_wiki, upload_track_lyrics

    # upload_album_wiki()
    # upload_track_wiki()
    # upload_track_lyrics()

    # gsheet_id = '1nO49CqcUj6_mBbPBi-liZ_UYolG2ylXdCfzVons6GnM'  # Album page
    # sheet_name = '28.09.2020'
    # gsheet_id = '15kQ54Ea7NYo-EoTkRhLC1-SxO4bAIodEFizely7FTkI'  # Single page
    # sheet_name = '28.09.2020'

    print("--- %s seconds ---" % (time.time() - start_time))
