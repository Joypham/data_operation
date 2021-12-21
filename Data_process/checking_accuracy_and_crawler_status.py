from Data_process.class_definition import WhenExist, PageType, SheetNames, merge_file, DataReports, \
    get_key_value_from_gsheet_info, add_key_value_from_gsheet_info, get_gsheet_id_from_url
from Data_process.youtube_similarity import similarity
from core.models.data_source_format_master import DataSourceFormatMaster

from core.models.crawlingtask_action_master import V4CrawlingTaskActionMaster
from core.crud.sql.query_supporter import get_crawlingtask_info, get_s11_crawlingtask_info, get_track_title_track_artist_by_ituneid_and_seq, get_youtube_crawlingtask_info
from crawl_itune.functions import get_itune_id_region_from_itune_url
from google_spreadsheet_api.function import update_value, update_value_at_last_column
from colorama import Fore, Style
import time
from core.crud.get_df_from_query import get_df_from_query
import pandas as pd


def checking_image_youtube_accuracy(df: object, actionid: str):
    df['check'] = ''
    df['status'] = ''
    df['crawlingtask_id'] = ''
    row_index = df.index
    for i in row_index:
        if actionid == V4CrawlingTaskActionMaster.ARTIST_ALBUM_IMAGE:
            objectid = df['uuid'].loc[i]
        elif actionid == V4CrawlingTaskActionMaster.DOWNLOAD_VIDEO_YOUTUBE:
            objectid = df['track_id'].loc[i]

        url = df.url_to_add.loc[i]
        gsheet_info = df.gsheet_info.loc[i]
        gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
        sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
        PIC_taskdetail = f"{gsheet_name}_{sheet_name}"
        db_crawlingtask = get_crawlingtask_info(objectid=objectid, PIC=PIC_taskdetail, actionid=actionid)

        if db_crawlingtask:
            status = db_crawlingtask.status
            crawlingtask_id = db_crawlingtask.id
            if url in db_crawlingtask.url:
                check_accuracy = True
            else:
                check_accuracy = f"crawlingtask_id: {db_crawlingtask.id}: uuid and url not match"
                print(check_accuracy)
        else:
            check_accuracy = f"file: {PIC_taskdetail}, uuid: {objectid} is missing"
            print(check_accuracy)
            status = 'missing'
            crawlingtask_id = 'missing'
        df.loc[i, 'check'] = check_accuracy
        df.loc[i, 'status'] = status
        df.loc[i, 'crawlingtask_id'] = crawlingtask_id
    return df


def automate_checking_status(df: object, actionid: str):
    gsheet_infos = list(set(df.gsheet_info.tolist()))
    count = 0
    while True and count < 300:
        checking_accuracy_result = checking_image_youtube_accuracy(df=df, actionid=actionid)
        result = checking_accuracy_result[
                     (checking_accuracy_result['status'] != 'complete')
                     & (checking_accuracy_result['status'] != 'incomplete')
                     ].status.tolist() == []
        if result == 1:
            for gsheet_info in gsheet_infos:
                gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
                sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
                print(
                    Fore.LIGHTYELLOW_EX + f"File: {gsheet_name}, sheet_name: {sheet_name} has been crawled complete already" + Style.RESET_ALL)
            break
        else:
            count += 1
            time.sleep(2)
            print(count, "-----", result)


def checking_s11_crawler_status(df: object):
    original_df = df.copy()
    original_df['itune_id'] = original_df['itune_album_url'].apply(
        lambda x: get_itune_id_region_from_itune_url(url=x)[0] if x not in (
            'None', '', 'not found', 'non', 'nan', 'Itunes_Album_Link') else 'None')
    original_df['url'] = original_df['gsheet_info'].apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))

    gsheet_infos = list(set(original_df.gsheet_info.tolist()))
    for gsheet_info in gsheet_infos:
        gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
        sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
        PIC_taskdetail = f"{gsheet_name}_{sheet_name}"
        url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
        original_df_split = original_df[original_df['url'] == url].reset_index()
        count = 0
        while True and count < 300:
            checking_accuracy_result = get_df_from_query(get_s11_crawlingtask_info(pic=PIC_taskdetail))
            checking_accuracy_result['itune_album_id'] = checking_accuracy_result['itune_album_id'].apply(
                lambda x: x.strip('"'))

            result = checking_accuracy_result[~
                                              ((checking_accuracy_result['06_status'] == 'complete')
                                               & (checking_accuracy_result['e5_status'] == 'complete')) |
                                              (checking_accuracy_result['06_status'] == 'incomplete') |
                                              ((checking_accuracy_result['06_status'] == 'complete')
                                               & (checking_accuracy_result['e5_status'] == 'incomplete'))
                                              ]

            checking = result.empty
            if checking == 1:
                print(
                    Fore.LIGHTYELLOW_EX + f"File: {gsheet_name}, sheet_name: {sheet_name} has been crawled complete already" + Style.RESET_ALL)
                data_merge = pd.merge(original_df_split, checking_accuracy_result, how='left', left_on='itune_id',
                                      right_on='itune_album_id', validate='m:1').fillna(value='None')
                print(data_merge)
                # update data to gsheet

                data_updated = data_merge[checking_accuracy_result.columns]
                update_value_at_last_column(df_to_update=data_updated, gsheet_id=get_gsheet_id_from_url(url=url),
                                            sheet_name=sheet_name)

                # update data report:
                data_report = data_merge[~
                ((
                         (data_merge['itune_album_url'].isin(['not found', '']))
                         & (data_merge['06_status'] == 'None')
                         & (data_merge['e5_status'] == 'None')
                 ) |
                 (
                         (~data_merge['itune_album_url'].isin(['not found', '']))
                         & (data_merge['06_status'] == 'complete')
                         & (data_merge['e5_status'] == 'complete')
                 ))
                ]
                if data_report.empty:
                    print(
                        Fore.LIGHTYELLOW_EX + f"Accuracy: ok\nStatus: ok" + Style.RESET_ALL)
                else:
                    print(
                        Fore.LIGHTYELLOW_EX + f"Accuracy: not ok\nStatus: not ok" + Style.RESET_ALL)
                    columns_data_report = ['itune_id'] + list(checking_accuracy_result.columns)
                    data_report = data_report[columns_data_report]
                    print(data_report)

                break
            else:
                count += 1
                print(
                    Fore.LIGHTYELLOW_EX + f"File: {gsheet_name}, sheet_name: {sheet_name} hasn't been crawled complete" + Style.RESET_ALL)
                time.sleep(10)
                print(count, "-----", result)


def get_format_id_from_content_type(content_type: str):
    if content_type in ("OFFICIAL_MUSIC_VIDEO", "OFFICIAL_MUSIC_VIDEO_2"):
        return DataSourceFormatMaster.FORMAT_ID_MP4_FULL
    elif content_type == "STATIC_IMAGE_VIDEO":
        return DataSourceFormatMaster.FORMAT_ID_MP3_FULL
    elif content_type == "COVER_VIDEO":
        return DataSourceFormatMaster.FORMAT_ID_MP4_COVER
    elif content_type == "LIVE_VIDEO":
        return DataSourceFormatMaster.FORMAT_ID_MP4_LIVE
    elif content_type == "REMIX_VIDEO":
        return DataSourceFormatMaster.FORMAT_ID_MP4_REMIX
    elif content_type == "LYRIC_VIDEO":
        return DataSourceFormatMaster.FORMAT_ID_MP4_LYRIC
    else:
        return "Unknown"


def checking_c11_crawler_status(original_df: object, pre_valid: str = None):
    original_df['itune_id'] = original_df.apply(
        lambda x: get_itune_id_region_from_itune_url(url=x['itune_album_url'])[0] if x['itune_album_url'] not in (
            'None', '', 'not found', 'non', 'nan', 'Itunes_Album_Link') else x['itune_id'], axis=1)
    original_df['url'] = original_df['gsheet_info'].apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))
    gsheet_infos = list(set(original_df.gsheet_info.tolist()))
    for gsheet_info in gsheet_infos:
        gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
        sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
        PIC_taskdetail = f"{gsheet_name}_{sheet_name}_{pre_valid}"
        url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
        original_df_split = original_df[original_df['url'] == url].reset_index()
        count = 0
        while True and count < 300:
            checking_accuracy_result = get_df_from_query(get_s11_crawlingtask_info(pic=PIC_taskdetail))
            checking_accuracy_result['itune_album_id'] = checking_accuracy_result['itune_album_id'].apply(
                lambda x: x.strip('"'))
            result = checking_accuracy_result[~
            (
                    ((checking_accuracy_result['06_status'] == 'complete')
                    & (checking_accuracy_result['E5_status'] == 'complete')) |

                    (checking_accuracy_result['06_status'] == 'incomplete') |

                    ((checking_accuracy_result['06_status'] == 'complete')
                    & (checking_accuracy_result['E5_status'] == 'incomplete')))
            ]
            checking = result.empty
            if checking == 1:
                print(
                    Fore.LIGHTYELLOW_EX + f"File: {gsheet_name}, sheet_name: {sheet_name} has been crawled complete already" + Style.RESET_ALL)

                data_merge = pd.merge(original_df_split, checking_accuracy_result, how='left', left_on='itune_id',
                                      right_on='itune_album_id', validate='m:1').fillna(value='None')
                data_merge['06_id_x'] = data_merge.apply(
                    lambda x: x['06_id_y'] if x['pre_valid'] == pre_valid else x['06_id_x'], axis=1)
                data_merge['06_status_x'] = data_merge.apply(
                    lambda x: x['06_status_y'] if x['pre_valid'] == pre_valid else x['06_status_x'], axis=1)
                data_merge['e5_id'] = data_merge.apply(
                    lambda x: x['E5_id'] if x['pre_valid'] == pre_valid else x['e5_id'], axis=1)
                data_merge['e5_status'] = data_merge.apply(
                    lambda x: x['E5_status'] if x['pre_valid'] == pre_valid else x['e5_status'], axis=1)
                data_merge.columns = data_merge.columns.str.replace('06_id_x', '06_id')
                data_merge.columns = data_merge.columns.str.replace('06_status_x', '06_status')
                data_merge = data_merge[original_df_split.columns]

                # update data report:
                data_report = data_merge[data_merge['pre_valid'] == pre_valid]

                data_report = data_report[~
                (
                        (
                                (data_report['itune_album_url'].isin(['not found', '']))
                                & (data_report['06_status'] == 'None')
                                & (data_report['e5_status'] == 'None')
                        )
                        |
                        (
                                (~data_report['itune_album_url'].isin(['not found', '']))
                                & (data_report['06_status'] == 'complete')
                                & (data_report['e5_status'] == 'complete')
                        )
                )
                ]
                if data_report.empty:
                    print(
                        Fore.LIGHTYELLOW_EX + f"Accuracy: ok\nStatus: ok" + Style.RESET_ALL)
                    row_num = data_merge.index
                    for i in row_num:
                        if data_merge['pre_valid'].loc[i] == pre_valid:
                            itune_album_id = data_merge['itune_id'].loc[i]
                            seq = data_merge['track_title/track_num'].loc[i]
                            format_id = get_format_id_from_content_type(content_type=data_merge['content type'].loc[i])
                            youtube_url = data_merge['contribution_link'].loc[i]
                            db_track = get_track_title_track_artist_by_ituneid_and_seq(itune_album_id=itune_album_id, seq=seq)
                            if db_track:
                                track_title = db_track.title
                                track_id = db_track.id
                                track_duration = db_track.duration_ms
                                track_similarity = similarity(track_title=track_title, youtube_url=youtube_url,
                                                              formatid=format_id,
                                                              duration=track_duration).get('similarity')
                            else:
                                track_title = 'not found'
                                track_id = 'not found'
                                track_similarity = 'not found'
                            data_merge.loc[i, 'track_title'] = track_title
                            data_merge.loc[i, 'track_id'] = track_id
                            data_merge.loc[i, 'similarity'] = track_similarity
                        else:
                            pass
                    updated_columns = ['06_id', '06_status', 'e5_id', 'e5_status', 'track_title', 'track_id', 'similarity']
                    print(data_merge[updated_columns])
                else:
                    print(
                        Fore.LIGHTYELLOW_EX + f"Accuracy: not ok\nStatus: not ok" + Style.RESET_ALL)
                    updated_columns = ['06_id', '06_status', 'e5_id', 'e5_status']
                # update data to gsheet
                data_updated = data_merge[updated_columns]
                grid_range_to_update = f"{sheet_name}!AM2"
                list_result = data_updated.values.tolist()  # transfer data_frame to 2D list
                update_value(list_result=list_result, grid_range_to_update=grid_range_to_update,
                             gsheet_id=get_gsheet_id_from_url(url=url))
                break
            else:
                count += 1
                print(
                    Fore.LIGHTYELLOW_EX + f"File: {gsheet_name}, sheet_name: {sheet_name} hasn't been crawled complete" + Style.RESET_ALL)
                time.sleep(10)
                print(count, "-----", result)


def checking_youtube_crawler_status(df: object, format_id: str):
    df['check'] = ''
    df['status'] = ''
    df['crawlingtask_id'] = ''
    row_index = df.index
    for i in row_index:
        url = df.url_to_add.loc[i]
        gsheet_info = df.gsheet_info.loc[i]
        gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
        sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
        PIC_taskdetail = f"{gsheet_name}_{sheet_name}"
        trackid = df.track_id.loc[i]
        url = df.url_to_add.loc[i]
        db_crawlingtask = get_youtube_crawlingtask_info(track_id=trackid, PIC=PIC_taskdetail, format_id= format_id)
        if db_crawlingtask:
            crawlingtask_id = db_crawlingtask.id
            status = db_crawlingtask.status
            db_url = db_crawlingtask.youtube_url.replace('"', '')
            check = (url == db_url)
        else:
            crawlingtask_id = "missing"
            status = "missing"
            check = "missing"
        df.loc[i, 'check'] = check
        df.loc[i, 'status'] = status
        df.loc[i, 'crawlingtask_id'] = crawlingtask_id
    return df


def automate_checking_youtube_crawler_status(original_df: object, filter_df: object, format_id: str):
    gsheet_infos = list(set(filter_df.gsheet_info.tolist()))
    count = 0
    while True and count < 300:
        checking_accuracy_result = checking_youtube_crawler_status(df=filter_df, format_id=format_id)
        result = checking_accuracy_result[
                     (checking_accuracy_result['status'] != 'complete')
                     & (checking_accuracy_result['status'] != 'incomplete')
                     & (checking_accuracy_result['status'] != 'missing')
                     ].status.tolist() == []
        if result == 1:
            for gsheet_info in gsheet_infos:
                gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
                sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
                url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
                print(
                    Fore.LIGHTYELLOW_EX + f"File: {gsheet_name}, sheet_name: {sheet_name} has been crawled complete already" + Style.RESET_ALL)
                # print(checking_accuracy_result)
                updated_column = ['check', 'status', 'crawlingtask_id']
                merge_df = original_df.merge(checking_accuracy_result[['check', 'status', 'crawlingtask_id', 'index']], left_index=True, right_on='index', how='left').fillna(value='')
                update_value_at_last_column(df_to_update=merge_df[updated_column],
                                            gsheet_id=get_gsheet_id_from_url(url=url), sheet_name=sheet_name)

            break
        else:
            count += 1
            time.sleep(2)
            print(count, "-----", result)


if __name__ == "__main__":
    start_time = time.time()

    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)