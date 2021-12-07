from core.models.data_source_format_master import DataSourceFormatMaster
import time
import pandas as pd
from core.models.crawlingtask_action_master import V4CrawlingTaskActionMaster
from Data_lake_process.class_definition import WhenExist, PageType, SheetNames, merge_file, Page, DataReports, \
    get_key_value_from_gsheet_info, add_key_value_from_gsheet_info, get_gsheet_id_from_url

from core.crud.sql.datasource import get_datasourceids_from_youtube_url_and_trackid, related_datasourceid
from itertools import chain
from core import query_path


class Mp3Type:
    C = 'c'
    D = 'd'
    Z = 'z'


def convert_dict(raw_dict: dict):
    keys_values = raw_dict.items()
    result = ""
    for key, value in keys_values:
        result = result + f"{key}: '{value}', "
    result = "{" + result[:-2] + "}"
    print(result)


def get_gsheet_id_from_url(url: str):
    url_list = url.split("/")
    gsheet_id = url_list[5]
    return gsheet_id


def crawl_itunes_album(ituneid: str, priority: int, is_new_release: bool = False, pic: str = "Joy_xinh", region: str = "us"):
    crawl_itunes_album = f"insert into crawlingtasks(Id, ActionId, TaskDetail, Priority) values (uuid4(), '{V4CrawlingTaskActionMaster.ITUNES_ALBUM}', JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.album_id', '{ituneid}', '$.region', '{region}', '$.PIC', '{pic}', '$.is_new_release', {is_new_release}), {priority});\n"
    return crawl_itunes_album


def crawl_youtube(track_id: str, youtube_url: str, format_id: str, when_exist: str = WhenExist.SKIP, place: str = None,
                  year: str = None, artist_cover: str = None,
                  pic: str = "Joy_xinh",
                  actionid: str = f"{V4CrawlingTaskActionMaster.DOWNLOAD_VIDEO_YOUTUBE}", priority: int = 1999):
    if artist_cover is not None:
        artist_cover = artist_cover.replace('\'', '\\\'').replace("\"", "\\\"")
    if place is not None:
        place = place.replace('\'', '\\\'').replace("\"", "\\\"")

    if format_id in (DataSourceFormatMaster.FORMAT_ID_MP4_FULL, DataSourceFormatMaster.FORMAT_ID_MP3_FULL,
                     DataSourceFormatMaster.FORMAT_ID_MP4_STATIC, DataSourceFormatMaster.FORMAT_ID_MP4_LYRIC):
        crawlingtask = f"insert into crawlingtasks(Id, ObjectID, ActionId, TaskDetail, Priority) values (uuid4(), '{track_id}', '{actionid}', JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.when_exists', '{when_exist}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.PIC', '{pic}'),{priority});\n"
    elif format_id in (DataSourceFormatMaster.FORMAT_ID_MP4_LIVE, DataSourceFormatMaster.FORMAT_ID_MP4_FAN_CAM):
        if place is None and year is None:
            crawlingtask = "-- missing info ---- concert_live_name or year"
        else:
            crawlingtask = f"insert into crawlingtasks(Id, ObjectID, ActionId, TaskDetail, Priority) values (uuid4(), '{track_id}', '{actionid}', JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.when_exists', '{when_exist}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.concert_live_name', '{place}','$.year', '{year}', '$.PIC', '{pic}'), {priority});\n"
    elif format_id == DataSourceFormatMaster.FORMAT_ID_MP4_COVER:
        if artist_cover is None:
            crawlingtask = "-- missing info ---- artist_cover"
        else:
            crawlingtask = f"insert into crawlingtasks(Id, ObjectID, ActionId, TaskDetail, Priority) values (uuid4(), '{track_id}', '{actionid}', JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.when_exists', '{when_exist}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.covered_artist_name', '{artist_cover}','$.year', '{year}', '$.PIC', '{pic}'), {priority});\n"
    else:
        crawlingtask = f"--formatid not support; \n"
    return crawlingtask


def crawl_image(objectid: str, url: str, object_type: str, when_exists: str = WhenExist.REPLACE, pic: str = "Joy_xinh",
                priority: int = 1999):
    crawl_image = f"insert into crawlingtasks(Id, ObjectID, ActionId, TaskDetail, Priority) values (uuid4(), '{objectid}', '{V4CrawlingTaskActionMaster.ARTIST_ALBUM_IMAGE}', JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.url', '{url}', '$.object_type', '{object_type}', '$.when_exists', '{when_exists}', '$.PIC', '{pic}'), {priority});\n"
    return crawl_image


def crawl_youtube_mp4(df: object):
    row_index = df.index
    with open(query_path, "a+") as f:
        for i in row_index:
            memo = df['memo'].loc[i]
            new_youtube_url = df['url_to_add'].loc[i]
            track_id = df['track_id'].loc[i]
            old_youtube_url = df['mp4_link'].loc[i]
            gsheet_name = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='gsheet_name')
            sheet_name_ = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='sheet_name')
            gsheet_id = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='gsheet_id')
            priority = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='page_priority')
            query = ""
            if memo == "not ok" and new_youtube_url == "none":
                datasourceids = get_datasourceids_from_youtube_url_and_trackid(youtube_url=old_youtube_url,
                                                                               trackid=track_id,
                                                                               formatid=DataSourceFormatMaster.FORMAT_ID_MP4_FULL).all()
                datasourceids_flatten_list = tuple(set(list(chain.from_iterable(datasourceids))))  # flatten list
                if datasourceids_flatten_list:
                    for datasourceid in datasourceids_flatten_list:
                        related_id_datasource = related_datasourceid(datasourceid)
                        if related_id_datasource == [(None, None, None)]:
                            query = query + f"Update datasources set valid = -10 where id = {datasourceid};\n"
                        else:
                            query = query + f"UPDATE datasources SET trackid = '', FormatID = ''  where id = '{datasourceid}';\n"
                            query = query + f"UPDATE datasources SET updatedAt = NOW() WHERE trackid = '{track_id}';\n"

                else:
                    query = query + f"-- not existed datasourceid searched by youtube_url: {old_youtube_url}, trackid:  {track_id}, format_id: {DataSourceFormatMaster.FORMAT_ID_MP4_FULL} in gssheet_name: {gsheet_name}, gsheet_id: {gsheet_id}, sheet_name: {sheet_name_} ;\n"

            elif memo == "not ok" and new_youtube_url != "none":
                query = query + crawl_youtube(track_id=track_id,
                                              youtube_url=new_youtube_url,
                                              format_id=DataSourceFormatMaster.FORMAT_ID_MP4_FULL,
                                              when_exist=WhenExist.REPLACE,
                                              priority=priority,
                                              pic=f"{gsheet_name}_{sheet_name_}"
                                              )
            elif memo == "added":
                query = query + crawl_youtube(track_id=track_id,
                                              youtube_url=new_youtube_url,
                                              format_id=DataSourceFormatMaster.FORMAT_ID_MP4_FULL,
                                              when_exist=WhenExist.SKIP,
                                              priority=priority,
                                              pic=f"{gsheet_name}_{sheet_name_}")
            f.write(query)


def crawl_youtube_mp3(df: object):
    row_index = df.index
    with open(query_path, "a+") as f:
        for i in row_index:
            memo = df['memo'].loc[i]
            new_youtube_url = df['url_to_add'].loc[i]
            track_id = df['track_id'].loc[i]
            old_youtube_url = df['mp3_link'].loc[i]
            gsheet_name = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='gsheet_name')
            sheet_name_ = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='sheet_name')
            gsheet_id = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='gsheet_id')
            priority = get_key_value_from_gsheet_info(gsheet_info=df['gsheet_info'].loc[i], key='page_priority')
            type = df['type'].loc[i]
            query = ""
            if memo == "not ok" and new_youtube_url == "none":
                datasourceids = get_datasourceids_from_youtube_url_and_trackid(youtube_url=old_youtube_url,
                                                                               trackid=track_id,
                                                                               formatid=DataSourceFormatMaster.FORMAT_ID_MP3_FULL).all()
                datasourceids_flatten_list = tuple(set(list(chain.from_iterable(datasourceids))))  # flatten list
                if datasourceids_flatten_list:
                    for datasourceid in datasourceids_flatten_list:
                        related_id_datasource = related_datasourceid(datasourceid)
                        if related_id_datasource == [(None, None, None)]:
                            query = query + f"Update datasources set valid = -10 where id = {datasourceid};\n"
                        else:
                            query = query + f"UPDATE datasources SET trackid = '', FormatID = ''  where id = '{datasourceid}';\n"
                            query = query + f"UPDATE datasources SET updatedAt = NOW() WHERE trackid = '{track_id}';\n"

                else:
                    query = query + f"-- not existed datasourceid searched by youtube_url: {old_youtube_url}, trackid:  {track_id}, format_id: {DataSourceFormatMaster.FORMAT_ID_MP3_FULL} in gssheet_name: {gsheet_name}, gsheet_id: {gsheet_id}, sheet_name: {sheet_name_} ;\n"

            elif memo == "not ok" and new_youtube_url != "none":
                query = query + crawl_youtube(track_id=track_id,
                                              youtube_url=new_youtube_url,
                                              format_id=DataSourceFormatMaster.FORMAT_ID_MP3_FULL,
                                              when_exist=WhenExist.REPLACE,
                                              priority=priority,
                                              pic=f"{gsheet_name}_{sheet_name_}"
                                              )
                if type.lower() in (Mp3Type.C, Mp3Type.Z):
                    query = query + crawl_youtube(track_id=track_id,
                                                  youtube_url=new_youtube_url,
                                                  format_id=DataSourceFormatMaster.FORMAT_ID_MP4_STATIC,
                                                  when_exist=WhenExist.SKIP,
                                                  priority=priority,
                                                  pic=f"{gsheet_name}_{sheet_name_}"
                                                  )
                elif type.lower() == Mp3Type.D:
                    query = query + crawl_youtube(track_id=track_id,
                                                  youtube_url=new_youtube_url,
                                                  format_id=DataSourceFormatMaster.FORMAT_ID_MP4_LYRIC,
                                                  when_exist=WhenExist.SKIP,
                                                  priority=priority,
                                                  pic=f"{gsheet_name}_{sheet_name_}"
                                                  )

            elif memo == "added":
                query = query + crawl_youtube(track_id=track_id,
                                              youtube_url=new_youtube_url,
                                              format_id=DataSourceFormatMaster.FORMAT_ID_MP3_FULL,
                                              when_exist=WhenExist.SKIP,
                                              priority=priority,
                                              pic=f"{gsheet_name}_{sheet_name_}")
                if type.lower() in (Mp3Type.C, Mp3Type.Z):
                    query = query + crawl_youtube(track_id=track_id,
                                                  youtube_url=new_youtube_url,
                                                  format_id=DataSourceFormatMaster.FORMAT_ID_MP4_STATIC,
                                                  when_exist=WhenExist.SKIP,
                                                  priority=priority,
                                                  pic=f"{gsheet_name}_{sheet_name_}"
                                                  )
                elif type.lower() == Mp3Type.D:
                    query = query + crawl_youtube(track_id=track_id,
                                                  youtube_url=new_youtube_url,
                                                  format_id=DataSourceFormatMaster.FORMAT_ID_MP4_LYRIC,
                                                  when_exist=WhenExist.SKIP,
                                                  priority=priority,
                                                  pic=f"{gsheet_name}_{sheet_name_}"
                                                  )
            f.write(query)


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


def update_contribution(pointlogsid: str, content_type: str, track_id: str, concert_live_name: str,
                        artist_name: str, year: str, pic: str, youtube_url: str, other_official_version: str):

    # format_id
    format_id = get_format_id_from_content_type(content_type=content_type)

    # when exist
    if content_type == 'OFFICIAL_MUSIC_VIDEO_2':
        when_exists = WhenExist.KEEP_BOTH
    elif format_id in (DataSourceFormatMaster.FORMAT_ID_MP4_FULL, DataSourceFormatMaster.FORMAT_ID_MP3_FULL):
        when_exists = WhenExist.SKIP
    else:
        when_exists = WhenExist.KEEP_BOTH

    if content_type == 'OFFICIAL_MUSIC_VIDEO_2':
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.other_official_version', '{other_official_version}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    elif content_type == "OFFICIAL_MUSIC_VIDEO":
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    elif content_type == 'STATIC_IMAGE_VIDEO':
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    elif content_type == 'LYRIC_VIDEO':
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    elif content_type == "REMIX_VIDEO":
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.remix_artist', '{artist_name}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    elif content_type == "COVER_VIDEO":
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.covered_artist_name', '{artist_name}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    elif content_type == "LIVE_VIDEO":
        query = f"UPDATE pointlogs SET VerifiedInfo = JSON_SET(IFNULL(pointlogs.VerifiedInfo, JSON_OBJECT()), '$.PIC', '{pic}','$.when_exists', '{when_exists}', '$.youtube_url', '{youtube_url}', '$.data_source_format_id', '{format_id}', '$.concert_live_name', '{concert_live_name}', '$.year', '{year}'), TargetId = '{track_id}', Valid = 1  WHERE id = '{pointlogsid}';"
    # elif content_type.contains('REJECT'):
    elif "REJECT" in content_type:
        query = f"UPDATE pointlogs SET  Valid = -2  WHERE id = '{pointlogsid}';"
    else:
        query = f"-- content_type not existed"
    return query


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 50, 'display.width', 1000)
    # content_type = 'STATIC_IMAGE_VIDEO'
    # track_id = 'EFBDFE26278B43B7ADBBCB4181A6267E'
    # live_concert_name_place = 'joy'
    # artist_name = 'joy'
    # year = 'joy'
    # pic = 'Justin_Contribution - 24032021'
    # year = ''
    # youtube_url = 'https://www.youtube.com/watch?v=OqBuXQLR4Y8https://youtu.be/OqBuXQLR4Y8https://youtu.be/OqBuXQLR4Y8https://youtu.be/OqBuXQLR4Y8https://youtu.be/OqBuXQLR4Y8'
    # point_log_id = '289894634B144A669F4C9F0A61947E03'

    # update_contribution(content_type=content_type, track_id=track_id, live_concert_name_place=live_concert_name_place, artist_name=artist_name, year=year, pic=pic,youtube_url=youtube_url, pointlogsid=point_log_id)
    url = "https://docs.google.com/spreadsheets/d/1ZUzx1smeyIKD4PtQ-hhT1kbPSTGRdu8I8NG1uvzcWr4/edit#gid=218846379&fvid=948579105"
    from google_spreadsheet_api.function import get_df_from_speadsheet
    df = get_df_from_speadsheet(gsheet_id=get_gsheet_id_from_url(url=url),sheet_name="Youtube collect_experiment")
    filter_df = df[((df['pre_valid'] == "2021-06-07") & (df['track_id'] != 'None'))]
    filter_df = df[

            ((df['pre_valid'] == '2021-06-07')
             & (~df['Content type'].str.contains('REJECT'))
             & (df['track_id'] != 'not found')
             )
    ].reset_index()
    filter_df = filter_df[['PointlogsID', 'Content type', 'OFFICIAL_MUSIC_VIDEO_2', 'Artist_Name', 'Year',
                     'Live_Concert_Name_Place', 'track_id', 'Contribution_link']]

    filter_df['crawling_task'] = filter_df.apply(
        lambda x: update_contribution(content_type=x['Content type'], track_id=x['track_id'], concert_live_name=x['Live_Concert_Name_Place'], artist_name=x['Artist_Name'], year=x['Year'], pic="joy xinh",youtube_url=x['Contribution_link'], pointlogsid=x['PointlogsID']), axis=1)
    # print(filter_df)
    row_num = filter_df.index
    for i in row_num:
        query = filter_df['crawling_task'].loc[i]
        print(query)
        # # conten_type = filter_df['Content type'].loc[i]
        # # conten_type = filter_df['Content type'].loc[i]
        # # conten_type = filter_df['Content type'].loc[i]
        # print(conten_type)

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
