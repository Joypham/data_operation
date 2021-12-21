from colorama import Fore, Style

from Data_process.class_definition import WhenExist, PageType, SheetNames, merge_file, Page, DataReports, \
    get_key_value_from_gsheet_info, add_key_value_from_gsheet_info, get_gsheet_id_from_url

from crawl_itune.functions import get_max_ratio, check_validate_itune, get_itune_id_region_from_itune_url
from google_spreadsheet_api.function import update_value, update_value_at_last_column


def youtube_check_box(page_name: str, df: object, sheet_name: str):
    df['len'] = df['url_to_add'].apply(lambda x: len(x))
    if page_name in ("TopSingle", "TopAlbum") and sheet_name == SheetNames.MP3_SHEET_NAME:

        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
                 & (df['type'].str.lower().isin(["c", "d", "z"]))
                 & (df['checking_mp3'] == 'TRUE')
                 & (df['already_existed'] == 'null')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == '')
                 & (df['url_to_add'] == '')
                 & (df['type'] == '')
                 & ~((df['checking_mp3'] == 'TRUE')
                     & (df['already_existed'] == 'null'))
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['len'] == 0)
                 & (df['checking_mp3'] == 'TRUE')
                 & (df['already_existed'] == 'null')
         )
         )
        ]
    elif page_name == "NewClassic" and sheet_name == SheetNames.MP3_SHEET_NAME:
        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
                 & (df['type'].str.lower().isin(["c", "d", "z"]))
                 & (df['checking_mp3'] == 'TRUE')
                 & (df['is_released'] == 'TRUE')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == '')
                 & (df['url_to_add'] == '')
                 & (df['type'] == '')
                 & ~((df['checking_mp3'] == 'TRUE')
                     & (df['is_released'] == 'TRUE'))
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['len'] == 0)
                 & (df['checking_mp3'] == 'TRUE')
                 & (df['is_released'] == 'TRUE')
         )
         )
        ]
    elif page_name == "TopSingle" and sheet_name == SheetNames.MP4_SHEET_NAME:

        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
                 & (df['checking_mp4'] == 'TRUE')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == '')
                 & (df['url_to_add'] == '')
                 & ~((df['checking_mp4'] == 'TRUE')
         )
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['len'] == 0)
                 & (df['checking_mp4'] == 'TRUE')
         )
         )
        ]
    elif page_name == "NewClassic" and sheet_name == SheetNames.MP4_SHEET_NAME:
        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
                 & (df['checking_mp4'] == 'TRUE')
                 & (df['is_released'] == 'TRUE')
                 & (df['verified'] == 'null')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == '')
                 & (df['url_to_add'] == '')
                 & ~(
                 (df['checking_mp4'] == 'TRUE')
                 & (df['is_released'] == 'TRUE')
                 & (df['is_released'] == 'null')
         )
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['len'] == 0)
                 & (df['checking_mp4'] == 'TRUE')
                 & (df['is_released'] == 'TRUE')
                 & (df['verified'] == 'null')
         )
         )
        ]
    elif page_name == "TopAlbum" and sheet_name == SheetNames.MP4_SHEET_NAME:
        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
                 & (df['checking_mp4'] == 'TRUE')
                 & (df['already_existed'] == 'null')
                 & (df['verified'] == 'null')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == '')
                 & (df['url_to_add'] == '')
                 & ~(
                 (df['checking_mp4'] == 'TRUE')
                 & (df['already_existed'] == 'null')
                 & (df['verified'] == 'null')
         )
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['len'] == 0)
                 & (df['checking_mp4'] == 'TRUE')
                 & (df['already_existed'] == 'null')
                 & (df['verified'] == 'null')
         )
         )
        ]
    elif page_name == "ArtistPage" and sheet_name == SheetNames.MP3_SHEET_NAME:
        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
                 & (df['type'].str.lower().isin(["c", "d", "z"]))
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['url_to_add'] == 'none')
                 & (df['type'] == 'none')
         ) |
         (

             (df['assignee'] == 'no need to check')
         ))
        ]
    elif page_name == "ArtistPage" and sheet_name == SheetNames.MP4_SHEET_NAME:
        youtube_check_box = df[~
        ((
                 (df['track_id'] != '')
                 & (df['memo'] == 'ok')
                 & (df['url_to_add'] == '')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'added')
                 & (df['len'] == 43)
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not found')
                 & (df['url_to_add'] == 'none')
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not ok')
                 & (df['len'] == 43)
         ) |
         (
                 (df['track_id'] != '')
                 & (df['memo'] == 'not ok')
                 & (df['url_to_add'] == 'none')
         ) |
         (df['assignee'] == 'no need to check')
         )
        ]
    if youtube_check_box.empty:
        print(Fore.LIGHTYELLOW_EX + f"Pass check box" + Style.RESET_ALL)
        return True
    else:
        print(Fore.LIGHTYELLOW_EX + f"Not pass check box" + Style.RESET_ALL)
        print(youtube_check_box)
        return False


def s11_checkbox(df: object):
    df['url'] = df['gsheet_info'].apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))
    # Step 1: check validate format
    check_format_s11 = df[~((df['itune_album_url'] == 'not found')
                            |(df['itune_album_url'].str[:32] == 'https://music.apple.com/us/album'))]

    if check_format_s11.empty:
        print(Fore.LIGHTYELLOW_EX + f"Pass check box" + Style.RESET_ALL)
        return True
    else:
        print(Fore.LIGHTYELLOW_EX + f"Not pass check box" + Style.RESET_ALL)
        print(check_format_s11.head(10))
        return False


def update_s11_check_box(df: object):
    gsheet_infos = list(set(df.gsheet_info.tolist()))
    df['url'] = df['gsheet_info'].apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))
    df['itune_id'] = df['itune_album_url'].apply(
        lambda x: get_itune_id_region_from_itune_url(url=x)[0] if x != 'not found' else 'None')
    df['region'] = df['itune_album_url'].apply(
        lambda x: get_itune_id_region_from_itune_url(url=x)[1] if x != 'not found' else 'None')
    df['checking_validate_itune'] = df['itune_id'].apply(
        lambda x: check_validate_itune(x) if x != 'None' else 'None')
    df['token_set_ratio'] = df.apply(
        lambda x: get_max_ratio(itune_album_id=x['itune_id'],
                                input_album_title=x['album_title']) if x['itune_id'] != 'None' else 'None',
        axis=1)
    # Update data
    for gsheet_info in gsheet_infos:
        url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
        df_to_upload = df[df['url'] == url].reset_index()
        # print(df_to_upload)
        column_name = ['itune_id', 'region', 'checking_validate_itune', 'token_set_ratio']
        updated_df = df_to_upload[column_name]
        update_value_at_last_column(df_to_update=updated_df, gsheet_id=get_gsheet_id_from_url(url),
                                    sheet_name=SheetNames.S_11)


def c11_checkbox(original_df: object, pre_valid: str = None):
    df = original_df[original_df['pre_valid'] == pre_valid].reset_index()
    df['itune_id'] = df['itune_album_url'].apply(
        lambda x: get_itune_id_region_from_itune_url(url=x)[0] if x not in (
            'None', '', 'not found', 'non', 'nan', 'Itunes_Album_Link') else 'None')
    df['url'] = df['gsheet_info'].apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))
    check_format_s11 = df[~((
                                    (~df['content type'].str.contains('REJECT'))
                                    & (df['itune_album_url'].str[:24] == 'https://music.apple.com/')
                            ) |
                            (
                                    (df['itune_album_url'] == '')
                                    & (df['content type'].str.contains('REJECT'))
                            ))]
    if check_format_s11.empty:
        print(Fore.LIGHTYELLOW_EX + f"Pass check box" + Style.RESET_ALL)
        return True
    else:
        print(Fore.LIGHTYELLOW_EX + f"Not pass check box" + Style.RESET_ALL)
        print(check_format_s11.head(10))
        return False


def update_c11_check_box(original_df: object, pre_valid: str):
    original_df['url'] = original_df.apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='url') if x[
                                                                                                 'pre_valid'] == pre_valid else 'None',
        axis=1)
    original_df['itune_id'] = original_df.apply(
        lambda x: get_itune_id_region_from_itune_url(url=x['itune_album_url'])[0] if (
                    x['itune_album_url'] != '' and x['pre_valid'] == pre_valid) else x['itune_id'], axis=1)
    original_df['region'] = original_df.apply(
        lambda x: get_itune_id_region_from_itune_url(url=x['itune_album_url'])[1] if (
                    x['itune_album_url'] != '' and x['pre_valid'] == pre_valid) else x['region'], axis=1)

    original_df['checking_validate_itune'] = original_df.apply(
        lambda x: check_validate_itune(itune_album_id=x['itune_id'], itune_region=x['region']) if (
                    x['itune_album_url'] != '' and x['pre_valid'] == pre_valid) else x['checking_validate_itune'],
        axis=1)

    gsheet_infos = list(set(original_df.gsheet_info.tolist()))
    sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_infos[0], key='sheet_name')
    url = get_key_value_from_gsheet_info(gsheet_info=gsheet_infos[0], key='url')
    grid_range_to_update = f"{sheet_name}!AJ2"
    list_result = original_df[
        ['itune_id', 'region', 'checking_validate_itune']].values.tolist()  # transfer data_frame to 2D list
    update_value(list_result=list_result, grid_range_to_update=grid_range_to_update,
                 gsheet_id=get_gsheet_id_from_url(url=url))
