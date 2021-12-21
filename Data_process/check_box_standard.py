from google_spreadsheet_api.function import get_df_from_speadsheet, get_list_of_sheet_title, update_value, \
    creat_new_sheet_and_update_data_from_df, get_gsheet_name
from Data_process.data_lake_standard import get_gsheet_id_from_url
from crawl_itune.functions import get_max_ratio, check_validate_itune, get_itune_id_region_from_itune_url

import pandas as pd
import time
from colorama import Fore, Style
from Data_process.data_lake_standard import update_data_reports


# S_11 = {"sheet_name": "S_11",
#         "column_name": ["release_date", "album_title", "album_artist", "itune_album_url", "sportify_album_url"]}

# Step 1: convert data into standard format:
def check_box_S_11_validate(gsheet_id: str):
    '''
    S_11 = {"sheet_name": "S_11",
            "column_name": ["release_date", "album_title", "album_artist", "itune_album_url", "sportify_album_url"]}
    '''

    sheet_info = sheet_type.S_11
    sheet_name = sheet_info.get('sheet_name')
    column_name = sheet_info.get('column_name')
    S_11_df = get_df_from_speadsheet(gsheet_id=gsheet_id, sheet_name=sheet_name)
    S_11_df.columns = S_11_df.columns.str.replace('Release_date', 'release_date')
    S_11_df.columns = S_11_df.columns.str.replace('AlbumTitle', 'album_title')
    S_11_df.columns = S_11_df.columns.str.replace('AlbumArtist', 'album_artist')
    S_11_df.columns = S_11_df.columns.str.replace('Itunes_Album_URL', 'itune_album_url')
    S_11_df.columns = S_11_df.columns.str.replace('AlbumURL', 'sportify_album_url')
    S_11_df = S_11_df[column_name].head(10)

    # Step 2: check validate format

    check_format_album_wiki = S_11_df[~((S_11_df['itune_album_url'] == 'not found')| (S_11_df['itune_album_url'].str[:32] == 'https://music.apple.com/us/album'))]
    S_11_format_validate = check_format_album_wiki.album_title.str.upper().to_numpy().tolist()
    if S_11_format_validate:
        print(check_format_album_wiki)
        return S_11_format_validate
    # Step 3: check validate itune_url
    else:
        S_11_df['itune_id'] = S_11_df['itune_album_url'].apply(
            lambda x: get_itune_id_region_from_itune_url(url=x)[0] if x != 'not found' else 'None')
        S_11_df['region'] = S_11_df['itune_album_url'].apply(
            lambda x: get_itune_id_region_from_itune_url(url=x)[1] if x != 'not found' else 'None')
        S_11_df['checking_validate_itune'] = S_11_df['itune_id'].apply(lambda x: check_validate_itune(x) if x != 'None' else 'None')
        S_11_df['token_set_ratio'] = S_11_df.apply(
            lambda x: get_max_ratio(itune_album_id=x['itune_id'], input_album_title=x['album_title']) if x['itune_id'] != 'None' else 'None', axis=1)

        # Step 4 update value:
        column_name = ['itune_id', 'region', 'checking_validate_itune', 'token_set_ratio']
        updated_df = S_11_df[column_name]

        list_result = updated_df.values.tolist()  # transfer data_frame to 2D list
        list_result.insert_column(0, column_name)
        range_to_update = f"{sheet_name}!M1"
        update_value(list_result, range_to_update,
                     gsheet_id)  # validate_value type: object, int, category... NOT DATETIME


def check_youtube_url_mp3(gsheet_id: str):
    '''
    MP3_SHEET_NAME = {"sheet_name": "MP_3", "fomatid": DataSourceFormatMaster.FORMAT_ID_MP3_FULL,
                      "column_name": ["track_id", "Memo", "Mp3_link", "url_to_add"]}
    '''
    sheet_info = sheet_type.MP3_SHEET_NAME
    sheet_name = sheet_info.get('sheet_name')
    column_name = sheet_info.get('column_name') + ['Type']
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)
    original_df.columns = original_df.columns.str.replace('TrackId', 'track_id')
    original_df.columns = original_df.columns.str.replace('MP3_link', 'Mp3_link')
    youtube_url_mp3 = original_df[column_name].copy()
    youtube_url_mp3['len'] = youtube_url_mp3['url_to_add'].apply(lambda x: len(x))

    check_youtube_url_mp3 = youtube_url_mp3[
    ((
             (youtube_url_mp3['track_id'] != '')
             & (youtube_url_mp3['Memo'] == 'added')
             & (youtube_url_mp3['len'] == 43)
             & (youtube_url_mp3['Type'].isin(["c", "d", "z"]))
     ) |
     (
             (youtube_url_mp3['track_id'] != '')
             & (youtube_url_mp3['Memo'] == 'not found')
             & (youtube_url_mp3['url_to_add'] == '')
             & (youtube_url_mp3['Type'] == '')
     ))
    ]
    print(check_youtube_url_mp3)
    # return check_youtube_url_mp3.track_id.str.upper()


def check_youtube_url_mp4(gsheet_id: str):
    '''
        TrackID	Memo	URL_to_add	Assignee
                                    no need to check
        not null	ok	null
        not null	added	length = 43
        not null	not found	none
        not null	not ok	length = 43
        not null	not ok	none
    :return:
    '''

    sheet_name = 'MP_4'
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)
    original_df['len'] = original_df['url_to_add'].apply(lambda x: len(x))
    youtube_url_mp4 = original_df[['track_id', 'Memo', 'url_to_add', 'len', 'Assignee']]

    check_youtube_url_mp4 = youtube_url_mp4[~
    ((
             (youtube_url_mp4['track_id'] != '')
             & (youtube_url_mp4['Memo'] == 'ok')
             & (youtube_url_mp4['url_to_add'] == '')
     ) |
     (
             (youtube_url_mp4['track_id'] != '')
             & (youtube_url_mp4['Memo'] == 'added')
             & (youtube_url_mp4['len'] == 43)
     ) |
     (
             (youtube_url_mp4['track_id'] != '')
             & (youtube_url_mp4['Memo'] == 'not found')
             & (youtube_url_mp4['url_to_add'] == 'none')
     ) |
     (
             (youtube_url_mp4['track_id'] != '')
             & (youtube_url_mp4['Memo'] == 'not ok')
             & (youtube_url_mp4['len'] == 43)
     ) |
     (
             (youtube_url_mp4['track_id'] != '')
             & (youtube_url_mp4['Memo'] == 'not ok')
             & (youtube_url_mp4['url_to_add'] == 'none')
     ) |
     (youtube_url_mp4['Assignee'] == 'no need to check')
     )
    ]
    return check_youtube_url_mp4.track_id.str.upper()


def check_version(gsheet_id: str):
    '''
    TrackID	    URL         	Remix_Artist
    not null	length = 43	    not null
    not null	null	        null

    TrackID	    URL2	        Venue           Live_year
    not null	length = 43	    not null        null hoặc nằm trong khoảng từ 1950 đến 2030
    not null	null	        null            null
    '''

    sheet_name = 'Version_done'
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)

    original_df['len_remix_url'] = original_df['Remix_url'].apply(lambda x: len(x))
    original_df['len_live_url'] = original_df['Live_url'].apply(lambda x: len(x))
    original_df['Live_year'] = pd.to_numeric(original_df.Live_year, errors='coerce').astype('Int64').fillna(0)

    youtube_url_version = original_df[
        ['track_id', 'Remix_url', 'Remix_artist', 'Live_url', 'Live_venue', 'len_remix_url', 'len_live_url',
         'Live_year']]

    check_version = youtube_url_version[~
    (((
              (youtube_url_version['track_id'] != '')
              & (youtube_url_version['len_remix_url'] == 43)
              & (youtube_url_version['Remix_artist'] != '')
      ) |
      (
              (youtube_url_version['track_id'] != '')
              & (youtube_url_version['Remix_url'] == '')
              & (youtube_url_version['Remix_artist'] == '')
      )) &
     ((
              (youtube_url_version['track_id'] != '')
              & (youtube_url_version['len_live_url'] == 43)
              & (youtube_url_version['Live_venue'] != '')
              & ((youtube_url_version['Live_year'] == 0) | (
              (1950 <= original_df['Live_year']) & (original_df['Live_year'] <= 2030)))
      ) |
      (
              (youtube_url_version['track_id'] != '')
              & (youtube_url_version['Live_url'] == '')
              & (youtube_url_version['Live_venue'] == '')
              & (youtube_url_version['Live_year'] == 0)
      )))
    ]
    return check_version.track_id.str.upper()


def check_album_image(gsheet_id: str):
    '''
        AlbumUUID	Memo	URL_to_add	Assignee
                                        no need to check
        not null	ok	null
        not null	added	not null
        not null	not found	none
    :return:
    '''
    sheet_name = 'Album_image'
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)
    album_image = original_df[['Album_uuid', 'Memo', 'url_to_add', 'Assignee']]

    check_album_image = album_image[~
    ((
             (album_image['Album_uuid'] != '')
             & (album_image['Memo'] == 'ok')
             & (album_image['url_to_add'] == '')
     ) |
     (
             (album_image['Album_uuid'] != '')
             & (album_image['Memo'] == 'added')
             & (album_image['url_to_add'] != '')
     ) |
     (
             (album_image['Album_uuid'] != '')
             & (album_image['Memo'] == 'not found')
             & (album_image['url_to_add'] == 'none')
     ) |
     (
         (album_image['Assignee'] == 'no need to check')
     ))
    ]

    return check_album_image.Album_uuid.str.upper()


def check_artist_image(gsheet_id: str):
    '''
        ArtistTrackUUID	Memo	URL_to_add	Assignee
                                            no need to check
        not null	ok	null
        not null	added	not null
    :return:
    '''
    sheet_name = 'Artist_image'
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)
    artist_image = original_df[['Artist_uuid', 'Memo', 'url_to_add', 'Assignee']]

    check_artist_image = artist_image[~
    ((
             (artist_image['Artist_uuid'] != '')
             & (artist_image['Memo'] == 'ok')
             & (artist_image['url_to_add'] == '')
     ) |
     (
             (artist_image['Artist_uuid'] != '')
             & (artist_image['Memo'] == 'added')
             & (artist_image['url_to_add'] != '')
     ) |
     (
         (artist_image['Assignee'] == 'no need to check')
     ))
    ]
    return check_artist_image.Artist_uuid.str.upper()


def check_album_wiki(gsheet_id: str):
    '''
        AlbumUUID	Memo	URL_to_add	Content_to_add	Assignee
                                                        no need to check
        not null	ok	null	null
        not null	added	https://en.wikipedia.org/%	not null
        not null	not found	none	none
        not null	not ok	https://en.wikipedia.org/%	not null
        not null	not ok	none	none
    :return:
    '''
    sheet_name = 'Album_wiki'
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)
    album_wiki = original_df[['Album_uuid', 'Memo', 'url_to_add', 'Content_to_add', 'Assignee']]

    check_album_wiki = album_wiki[~
    ((
             (album_wiki['Album_uuid'] != '')
             & (album_wiki['Memo'] == 'ok')
             & (album_wiki['url_to_add'] == '')
             & (album_wiki['Content_to_add'] == '')
     ) |
     (
             (album_wiki['Album_uuid'] != '')
             & (album_wiki['Memo'] == 'added')
             & (album_wiki['url_to_add'].str[:25] == 'https://en.wikipedia.org/')
             & (album_wiki['Content_to_add'] != '')
     ) |
     (
             (album_wiki['Album_uuid'] != '')
             & (album_wiki['Memo'] == 'not found')
             & (album_wiki['url_to_add'] == 'none')
             & (album_wiki['Content_to_add'] == 'none')
     ) |
     (
             (album_wiki['Album_uuid'] != '')
             & (album_wiki['Memo'] == 'not ok')
             & (album_wiki['url_to_add'].str[:25] == 'https://en.wikipedia.org/')
             & (album_wiki['Content_to_add'] != '')
     ) |
     (
             (album_wiki['Album_uuid'] != '')
             & (album_wiki['Memo'] == 'not ok')
             & (album_wiki['url_to_add'] == 'none')
             & (album_wiki['Content_to_add'] == 'none')
     ) |
     (
         (album_wiki['Assignee'] == 'no need to check')
     ))
    ]
    return check_album_wiki.Album_uuid.str.upper()


def check_artist_wiki(gsheet_id: str):
    '''
        Artist_uuid	Memo	URL_to_add	Assignee
                    no need to check
        not null	ok	null
        not null	added	https://en.wikipedia.org/%
        not null	not found	none
        not null	not ok	https://en.wikipedia.org/%
        not null	not ok	none
    :return:
    '''

    sheet_name = 'Artist_wiki'
    original_df = get_df_from_speadsheet(gsheet_id, sheet_name).applymap(str.lower)
    artist_wiki = original_df[['Artist_uuid', 'Memo', 'url_to_add', 'Assignee']]

    check_artist_wiki = artist_wiki[~
    ((
             (artist_wiki['Artist_uuid'] != '')
             & (artist_wiki['Memo'] == 'ok')
             & (artist_wiki['url_to_add'] == '')
     ) |
     (
             (artist_wiki['Artist_uuid'] != '')
             & (artist_wiki['Memo'] == 'added')
             & (artist_wiki['url_to_add'].str[:25] == 'https://en.wikipedia.org/')
     ) |
     (
             (artist_wiki['Artist_uuid'] != '')
             & (artist_wiki['Memo'] == 'not found')
             & (artist_wiki['url_to_add'] == 'none')
     ) |
     (
             (artist_wiki['Artist_uuid'] != '')
             & (artist_wiki['Memo'] == 'not ok')
             & (artist_wiki['url_to_add'].str[:25] == 'https://en.wikipedia.org/')
     ) |
     (
             (artist_wiki['Artist_uuid'] != '')
             & (artist_wiki['Memo'] == 'not ok')
             & (artist_wiki['url_to_add'] == 'none')
     ) |
     (
         (artist_wiki['Assignee'] == 'no need to check')
     ))
    ]
    return check_artist_wiki.Artist_uuid.str.upper()


def check_box(urls: list):
    for url in urls:
        gsheet_id = get_gsheet_id_from_url(url)
        gsheet_name = get_gsheet_name(gsheet_id=gsheet_id)
        print(Fore.LIGHTYELLOW_EX + f" \n{gsheet_name}\n{url}\n" + Style.RESET_ALL)
        # Step 1: check all elements
        list_of_sheet_title = get_list_of_sheet_title(gsheet_id)
        if 'MP_3' in list_of_sheet_title:
            youtube_url_mp3 = check_youtube_url_mp3(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            youtube_url_mp3 = ["MP_3 not found"]

        if 'MP_4' in list_of_sheet_title:
            youtube_url_mp4 = check_youtube_url_mp4(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            youtube_url_mp4 = ["MP_4 not found"]

        if 'Version_done' in list_of_sheet_title:
            version = check_version(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            version = ["Version_done not found"]

        if 'Album_image' in list_of_sheet_title:
            album_image = check_album_image(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            album_image = ["Album_image not found"]

        if 'Artist_image' in list_of_sheet_title:
            artist_image = check_artist_image(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            artist_image = ["Artist_image not found"]

        if 'Album_wiki' in list_of_sheet_title:
            album_wiki = check_album_wiki(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            album_wiki = ["Album_wiki not found"]

        if 'Artist_wiki' in list_of_sheet_title:
            artist_wiki = check_artist_wiki(gsheet_id=gsheet_id).to_numpy().tolist()
        else:
            artist_wiki = ["Artist_wiki not found"]

        # Step 2: create check_box
        sheet_name = []
        status = []
        comment = []

        dict_value = [youtube_url_mp3, youtube_url_mp4, version, album_image, artist_image, album_wiki, artist_wiki]
        dict_key = ['MP_3', 'MP_4', 'Version_done', 'Album_image', 'Artist_image', 'Album_wiki',
                    'Artist_wiki']
        dict_result = dict(zip(dict_key, dict_value))
        for i, j in dict_result.items():
            if not j:
                status.append('ok')
                comment.append(None)
                sheet_name.append(i)
            else:
                status.append('not ok')
                comment.append(j)
                sheet_name.append(i)
        d = {'sheet_name': sheet_name, 'status': status, 'comment': comment}
        check_box = pd.DataFrame(data=d).astype(str)
        print(check_box)
        creat_new_sheet_and_update_data_from_df(check_box, gsheet_id, 'jane_to_check_result')

        # Step 3: update data_report if meet conditions

        check_box_filtered = check_box[~(
                (check_box['status'] == "not ok")
                & (check_box['comment'].str.contains('not found')))
        ]
        checking = 'not ok' in check_box_filtered.status.drop_duplicates().tolist()
        if checking == 1:
            print("Please recheck check_box")
        else:
            list_sheet_name = check_box_filtered['sheet_name'].tolist()
            for sheet_name in list_sheet_name:
                gsheet_info = str({'url': f'{url}', 'gsheet_id': f'{gsheet_id}',
                                   'gsheet_name': f'{get_gsheet_name(gsheet_id=gsheet_id)}',
                                   'sheet_name': f'{sheet_name}'})
                print(gsheet_info)
                update_data_reports(gsheet_info=gsheet_info, notice="check_box completed")

        return check_box


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 50, 'display.width', 1000)
    urls = [
        "https://docs.google.com/spreadsheets/d/1W2QmYccbfeEAOEboKGSFWhv9hsXoQGPSZUhMP9Njsfw/edit#gid=1941765562",
        # "https://docs.google.com/spreadsheets/d/15LL8rcVnsWjE7D4RvrIMRpH8Y9Lgyio-kcs4mE540MI/edit#gid=1308575784"
    ]
    for url in urls:
        gsheet_id = get_gsheet_id_from_url(url)
        k = get_list_of_sheet_title(gsheet_id)
        check_youtube_url_mp3(gsheet_id=gsheet_id)

        # print(k)

    # k = check_youtube_url_mp3(gsheet_id="1W2QmYccbfeEAOEboKGSFWhv9hsXoQGPSZUhMP9Njsfw", sheet_info=sheet_info)
    print("--- %s seconds ---" % (time.time() - start_time))
