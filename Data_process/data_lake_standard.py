from google_spreadsheet_api.function import get_df_from_speadsheet, creat_new_sheet_and_update_data_from_df, \
    get_gsheet_name
from core.models.crawlingtask_action_master import V4CrawlingTaskActionMaster
from core.models.data_source_format_master import DataSourceFormatMaster
from core.crud.sql import artist, album
from core.crud.sql.track import get_one_track_by_id
import pandas as pd
import time
from core import query_path
from colorama import Fore, Style
from Data_process.crawlingtask import crawl_image, crawl_youtube_mp3, crawl_youtube_mp4, crawl_itunes_album, \
    update_contribution
from Data_process.class_definition import WhenExist, PageType, SheetNames, merge_file, DataReports, \
    get_key_value_from_gsheet_info, add_key_value_from_gsheet_info, get_gsheet_id_from_url
from Data_process.new_check_box_standard import youtube_check_box, s11_checkbox, update_s11_check_box, \
    c11_checkbox, update_c11_check_box
from Data_process.data_report import update_data_reports
from Data_process.checking_accuracy_and_crawler_status import checking_image_youtube_accuracy, \
    automate_checking_status, checking_s11_crawler_status, checking_c11_crawler_status, checking_youtube_crawler_status, \
    automate_checking_youtube_crawler_status
from crawl_itune.functions import get_itune_id_region_from_itune_url
from core.crud.get_df_from_query import get_df_from_query
from core.crud.sql.query_supporter import get_pointlogsid_valid, get_youtube_crawlingtask_info
from google_spreadsheet_api.function import update_value, update_value_at_last_column
from Data_process.class_definition import get_gsheet_id_from_url
from datetime import date
from Data_process.youtube_similarity import similarity


def upload_image_cant_crawl(checking_accuracy_result: object, sheet_name: str):
    gsheet_infos = list(set(checking_accuracy_result.gsheet_info.tolist()))
    df_incomplete = checking_accuracy_result[(checking_accuracy_result['status'] == 'incomplete')].reset_index().copy()

    df_incomplete['url'] = df_incomplete['gsheet_info'].apply(
        lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))
    df_incomplete['url_to_add'] = ''
    if sheet_name == SheetNames.ARTIST_IMAGE:
        df_incomplete['name'] = df_incomplete['uuid'].apply(
            lambda x: artist.get_one_by_id(artist_uuid=x).name)
    else:
        df_incomplete['title'] = df_incomplete['uuid'].apply(lambda x: artist.get_one_by_id(artist_uuid=x).title)
        df_incomplete['artist'] = df_incomplete['uuid'].apply(lambda x: album.get_one_by_id(album_uuid=x).artist)

    df_incomplete = df_incomplete[
        ['uuid', 'name', 'status', 'crawlingtask_id', 'url', 'memo', 'url_to_add']]

    for gsheet_info in gsheet_infos:
        url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
        df_incomplete_to_upload = df_incomplete[df_incomplete['url'] == url].reset_index()
        count_incomplete = df_incomplete_to_upload.index.n_estimators_stop
        joy = df_incomplete_to_upload['status'].tolist() == []

        if joy:
            raw_df_to_upload = {'status': ['Upload thành công 100% nhé các em ^ - ^']}
            df_to_upload = pd.DataFrame(data=raw_df_to_upload)
        else:
            df_to_upload = df_incomplete_to_upload.drop(['url', 'index'], axis=1)
        new_sheet_name = f"{sheet_name_} cant upload"
        print(df_to_upload)
        creat_new_sheet_and_update_data_from_df(df_to_upload, get_gsheet_id_from_url(url), new_sheet_name)


def query_pandas_to_csv(df: object, column: str):
    row_index = df.index
    with open(query_path, "w") as f:
        for i in row_index:
            line = df[column].loc[i]
            f.write(line)
    f.close()


class ImageWorking:
    def __init__(self, sheet_name: str, urls: list, page_type: object):
        original_file_ = merge_file(sheet_name=sheet_name, urls=urls, page_type=page_type)
        if original_file_.empty:
            print("original_file is empty")
            pass
        else:
            self.original_file = original_file_
            self.sheet_name = sheet_name
            self.page_type = page_type

    def check_box(self):
        print(self.original_file.head(10))

    def image_filter(self):
        df = self.original_file
        filter_df = df[((df['memo'] == 'missing') | (df['memo'] == 'added'))  # filter df by conditions
                       & (df['url_to_add'].notnull())
                       & (df['url_to_add'] != '')
                       ].drop_duplicates(subset=['uuid', 'url_to_add', 'gsheet_info'], keep='first').reset_index()

        if self.sheet_name == SheetNames.ARTIST_IMAGE:
            object_type_ = {"object_type": "artist"}
            filter_df['gsheet_info'] = filter_df.apply(
                lambda x: add_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key_value=object_type_), axis=1)

        elif self.sheet_name == SheetNames.ALBUM_IMAGE:
            object_type_ = {"object_type": "album"}
            filter_df['gsheet_info'] = filter_df['gsheet_info'].apply(
                lambda x: add_key_value_from_gsheet_info(gsheet_info=x, key_value=object_type_))
        else:
            pass
        return filter_df

    def crawl_image_datalake(self, when_exists: str = WhenExist.REPLACE):
        df = self.image_filter()
        if df.empty:
            print(Fore.LIGHTYELLOW_EX + f"Image file is empty" + Style.RESET_ALL)
        else:
            df['query'] = df.apply(lambda x:
                                   crawl_image(
                                       object_type=get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'],
                                                                                  key='object_type'),
                                       url=x['url_to_add'],
                                       objectid=x['uuid'],
                                       when_exists=when_exists,
                                       pic=f"{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='gsheet_name')}_{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='sheet_name')}",
                                       priority=get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'],
                                                                               key='page_priority')
                                   ),
                                   axis=1)
            query_pandas_to_csv(df=df, column='query')

    def checking_image_crawler_status(self):
        print("checking accuracy")
        df = self.image_filter().copy()
        gsheet_infos = list(set(df.gsheet_info.tolist()))
        # step 1.1: checking accuracy
        checking_accuracy_result = checking_image_youtube_accuracy(df=df,
                                                                   actionid=V4CrawlingTaskActionMaster.ARTIST_ALBUM_IMAGE)
        accuracy_checking = list(set(checking_accuracy_result['check'].tolist()))

        if accuracy_checking != [True]:
            print(checking_accuracy_result[['uuid', 'check', 'status', 'crawlingtask_id']])
            # Step 1.2: update data_reports if checking accuracy fail
            for gsheet_info in gsheet_infos:
                update_data_reports(gsheet_info=gsheet_info, status=DataReports.status_type_processing,
                                    notice="check accuracy fail")
        # Step 2: auto checking status
        else:
            print("checking accuracy correctly, now checking status")
            automate_checking_status(df=df, actionid=V4CrawlingTaskActionMaster.ARTIST_ALBUM_IMAGE)
            # Step 3: upload image cant crawl
            upload_image_cant_crawl(checking_accuracy_result=checking_accuracy_result, sheet_name=self.sheet_name)


class YoutubeWorking:
    def __init__(self, sheet_name: str, urls: list, page_type: object):
        original_file_ = merge_file(sheet_name=sheet_name, urls=urls, page_type=page_type)
        if original_file_.empty:
            print("original_file is empty")
            pass
        else:
            self.original_file = original_file_
            self.sheet_name = sheet_name
            self.page_type = page_type

    def check_box(self):
        df = self.original_file
        youtube_check_box(page_name=getattr(self.page_type, "name"), df=df, sheet_name=self.sheet_name)
        return youtube_check_box

    def youtube_filter(self):
        if self.check_box():
            df = self.original_file
            if self.sheet_name == SheetNames.MP3_SHEET_NAME:
                filter_df = df[((df['memo'] == 'not ok') | (df['memo'] == 'added'))  # filter df by conditions
                               & (df['url_to_add'].notnull())
                               & (df['url_to_add'] != '')
                               ].drop_duplicates(subset=['track_id', 'url_to_add', 'type', 'gsheet_info'],
                                                 keep='first').reset_index()
            elif self.sheet_name == SheetNames.MP4_SHEET_NAME:
                filter_df = df[((df['memo'] == 'not ok') | (df['memo'] == 'added'))  # filter df by conditions
                               & (df['url_to_add'].notnull())
                               & (df['url_to_add'] != '')
                               ].drop_duplicates(subset=['track_id', 'url_to_add', 'gsheet_info'],
                                                 keep='first').reset_index()
            return filter_df

    def crawl_mp3_mp4_youtube_datalake(self):
        df = self.youtube_filter()
        if self.sheet_name == SheetNames.MP3_SHEET_NAME:
            crawl_youtube_mp3(df=df)
        elif self.sheet_name == SheetNames.MP4_SHEET_NAME:
            crawl_youtube_mp4(df=df)
        else:
            pass

    def checking_youtube_crawler_status(self):
        print("checking accuracy")
        original_df = self.original_file
        filter_df = self.youtube_filter()
        if self.sheet_name == SheetNames.MP3_SHEET_NAME:
            format_id = DataSourceFormatMaster.FORMAT_ID_MP3_FULL
        elif self.sheet_name == SheetNames.MP4_SHEET_NAME:
            format_id = DataSourceFormatMaster.FORMAT_ID_MP4_FULL
        else:
            print("format_id not support")
            pass
        automate_checking_youtube_crawler_status(original_df=self.original_file, filter_df=self.youtube_filter().copy(),
                                                 format_id=format_id)

    def similarity(self):
        df = self.original_file
        df['similarity'] = ''
        df['note'] = ''
        if self.sheet_name == SheetNames.MP3_SHEET_NAME:
            format_id = DataSourceFormatMaster.FORMAT_ID_MP3_FULL
        elif self.sheet_name == SheetNames.MP4_SHEET_NAME:
            format_id = DataSourceFormatMaster.FORMAT_ID_MP4_FULL
        else:
            print("format_id not support")
            pass
        gsheet_info = list(set(df.gsheet_info.tolist()))[0]
        sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
        url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
        row_num = df.index
        for i in row_num:
            if df['memo'].loc[i] == 'added':
                trackid = df['track_id'].loc[i]
                youtube_url = df['url_to_add'].loc[i]
                db_track = get_one_track_by_id(track_id=trackid)
                if db_track:
                    track_title = db_track.title
                    track_duration = db_track.duration_ms
                    track_similarity = similarity(track_title=track_title, youtube_url=youtube_url,
                                                  formatid=format_id,
                                                  duration=track_duration).get('similarity')
                else:
                    track_similarity = 'not found'
                df.loc[i, 'similarity'] = track_similarity
            else:
                pass

        update_value_at_last_column(df_to_update=df[['similarity', 'note']], gsheet_id=get_gsheet_id_from_url(url=url),
                                    sheet_name=sheet_name)


class S11Working:
    def __init__(self, sheet_name: str, urls: list, page_type: object):
        original_file_ = merge_file(sheet_name=sheet_name, urls=urls, page_type=page_type)
        if original_file_.empty:
            print("original_file is empty")
            pass
        else:
            self.original_file = original_file_
            self.sheet_name = sheet_name
            self.page_type = page_type

    def check_box(self):
        df = self.original_file
        s11_checkbox(df=df)
        update_s11_check_box(df=df)

    def s11_filter(self):
        df = self.original_file
        s11_checkbox(df=df)
        if s11_checkbox(df=df):
            filter_df = df[
                (df['itune_album_url'] != 'not found') & (df['itune_album_url'] != '')
                ].drop_duplicates(
                subset=['itune_album_url', 'gsheet_info'], keep='first').reset_index()

            filter_df['itune_id'] = filter_df['itune_album_url'].apply(
                lambda x: get_itune_id_region_from_itune_url(url=x)[0])
            filter_df['region'] = filter_df['itune_album_url'].apply(
                lambda x: get_itune_id_region_from_itune_url(url=x)[1])
            return filter_df
        else:
            pass

    def crawl_s11_datalake(self):
        df = self.s11_filter()
        if getattr(self.page_type, "name") == "NewClassic":
            is_new_release = True
        else:
            is_new_release = False
        if df.empty:
            print(Fore.LIGHTYELLOW_EX + f"s11 file is empty" + Style.RESET_ALL)
        else:
            df['query'] = df.apply(lambda x:
                                   crawl_itunes_album(ituneid=x['itune_id'],
                                                      priority=get_key_value_from_gsheet_info(
                                                          gsheet_info=x['gsheet_info'], key='page_priority'),
                                                      is_new_release=is_new_release,
                                                      pic=f"{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='gsheet_name')}_{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='sheet_name')}",
                                                      region=x['region']
                                                      ),
                                   axis=1)
        query_pandas_to_csv(df=df, column='query')

    def checking_s11_crawler_status(self):
        checking_s11_crawler_status(df=self.original_file)


class C11Working:
    def __init__(self, sheet_name: str, urls: list, page_type: object, pre_valid: str = None):
        original_file_ = merge_file(sheet_name=sheet_name, urls=urls, page_type=page_type)
        if original_file_.empty:
            print("original_file is empty")
            pass
        else:
            self.original_file = original_file_
            self.sheet_name = sheet_name
            self.page_type = page_type
            self.pre_valid = pre_valid

    def pre_valid_(self):
        original_df = self.original_file
        original_df['url'] = original_df['gsheet_info'].apply(
            lambda x: get_key_value_from_gsheet_info(gsheet_info=x, key='url'))
        gsheet_infos = list(set(original_df.gsheet_info.tolist()))
        for gsheet_info in gsheet_infos:
            url = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='url')
            original_df_split = original_df[original_df['url'] == url]
            pointlogids = original_df_split[original_df_split['pointlogsid'] != '']['pointlogsid'].tolist()
            pointlogids_prevalid = get_df_from_query(get_pointlogsid_valid(pointlogids=pointlogids))
            data_merge = pd.merge(original_df_split, pointlogids_prevalid, how='left', left_on='pointlogsid',
                                  right_on='id', validate='m:1').fillna(value='None')
            data_merge = data_merge[data_merge['id'] != 'None']
            row_index = data_merge.index
            for i in row_index:
                range_to_update = f"Youtube collect_experiment!A{i + 2}"
                current_date = f"{date.today()}"
                list_result = [[current_date]]
                update_value(list_result=list_result, grid_range_to_update=range_to_update,
                             gsheet_id=get_gsheet_id_from_url(url))

    def check_box(self):
        df = self.original_file
        c11_checkbox(original_df=df, pre_valid=self.pre_valid)
        update_c11_check_box(original_df=df, pre_valid=self.pre_valid)

    def c11_filter(self):
        df = self.original_file
        if c11_checkbox(original_df=df, pre_valid=self.pre_valid):
            filter_df = df[
                (df['pre_valid'] == self.pre_valid)
                & (df['itune_album_url'] != '')
                & (~df['content type'].str.contains('REJECT'))
                ].reset_index()
            filter_df['itune_id'] = filter_df['itune_album_url'].apply(
                lambda x: get_itune_id_region_from_itune_url(url=x)[0])
            filter_df['region'] = filter_df['itune_album_url'].apply(
                lambda x: get_itune_id_region_from_itune_url(url=x)[1])
            filter_df = filter_df.drop_duplicates(subset=['itune_id', 'gsheet_info'], keep='first').reset_index()
            return filter_df

    def crawl_c11_datalake(self):
        df = self.c11_filter()
        if getattr(self.page_type, "name") == "NewClassic":
            is_new_release = True
        else:
            is_new_release = False
        if df.empty:
            print(Fore.LIGHTYELLOW_EX + f"s11 file is empty" + Style.RESET_ALL)
        else:
            df['query'] = df.apply(lambda x:
                                   crawl_itunes_album(ituneid=x['itune_id'],
                                                      priority=get_key_value_from_gsheet_info(
                                                          gsheet_info=x['gsheet_info'], key='page_priority'),
                                                      is_new_release=is_new_release,
                                                      pic=f"{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='gsheet_name')}_{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='sheet_name')}_{x['pre_valid']}",
                                                      region=x['region']
                                                      ),
                                   axis=1)
        query_pandas_to_csv(df=df, column='query')

    def checking_c11_crawler_status(self):
        checking_c11_crawler_status(original_df=self.original_file, pre_valid=self.pre_valid)

    def update_d9(self):
        filter_df = self.original_file
        filter_df = filter_df[
            ((filter_df['pre_valid'] == pre_valid)
                # & (~filter_df['content type'].str.contains('REJECT'))
                # & (filter_df['track_id'] != 'not found')
                )
        ].reset_index()
        gsheet_info = list(set(filter_df.gsheet_info.tolist()))[0]
        gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
        sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
        PIC_taskdetail = f"{gsheet_name}_{sheet_name}_{pre_valid}"
        filter_df['crawling_task'] = filter_df.apply(
            lambda x: update_contribution(content_type=x['content type'], track_id=x['track_id'],
                                          concert_live_name=x['live_concert_name_place'], artist_name=x['artist_name'],
                                          year=x['year'], pic=PIC_taskdetail, youtube_url=x['contribution_link'],
                                          other_official_version=x['official_music_video_2'],
                                          pointlogsid=x['pointlogsid']), axis=1)

        row_index = filter_df.index
        with open(query_path, "w") as f:
            for i in row_index:
                line = filter_df['crawling_task'].loc[i]
                # print(line)
                f.write(f"{line}\n")
        f.close()


class ControlFlow:
    def __init__(self, sheet_name: str, urls: list, page_type: object, pre_valid: str = ""):
        self.page_type = page_type
        self.urls = urls
        self.sheet_name = sheet_name
        self.pre_valid = pre_valid

    def pre_valid_(self):
        if self.sheet_name == SheetNames.C_11:
            c11_working = C11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type,
                                     pre_valid=self.pre_valid)
            return c11_working.pre_valid_()

    def update_d9(self):
        if self.sheet_name == SheetNames.C_11:
            c11_working = C11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type,
                                     pre_valid=self.pre_valid)
            return c11_working.update_d9()

    def check_box(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return image_working.check_box()
        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return youtube_working.check_box()
        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return s11_working.check_box()
        elif self.sheet_name == SheetNames.C_11:
            c11_working = C11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type,
                                     pre_valid=self.pre_valid)
            return c11_working.check_box()

    def observe(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return image_working.image_filter()
        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return youtube_working.youtube_filter()
        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return s11_working.s11_filter()
        elif self.sheet_name == SheetNames.C_11:
            c11_working = C11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type,
                                     pre_valid=self.pre_valid)
            return c11_working.c11_filter()

    def crawl(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return image_working.crawl_image_datalake()

        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return youtube_working.crawl_mp3_mp4_youtube_datalake()

        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return s11_working.crawl_s11_datalake()

        elif self.sheet_name == SheetNames.C_11:
            c11_working = C11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type,
                                     pre_valid=self.pre_valid)
            return c11_working.crawl_c11_datalake()

    def checking(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return image_working.checking_image_crawler_status()

        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return youtube_working.checking_youtube_crawler_status()

        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return s11_working.checking_s11_crawler_status()

        elif self.sheet_name == SheetNames.C_11:
            c11_working = C11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type,
                                     pre_valid=self.pre_valid)
            return c11_working.checking_c11_crawler_status()

    def similarity(self):
        if self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return youtube_working.similarity()


if __name__ == "__main__":
    start_time = time.time()

    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    with open(query_path, "w") as f:
        f.truncate()
    urls = [
        # "https://docs.google.com/spreadsheets/d/1ZUzx1smeyIKD4PtQ-hhT1kbPSTGRdu8I8NG1uvzcWr4/edit#gid=218846379"
        # "https://docs.google.com/spreadsheets/d/11SWGQ8AYGq65CbUotKfEVq_ZCGoIt32BpytxAx5z3M0/edit#gid=0"
        # "https://docs.google.com/spreadsheets/d/1ZUzx1smeyIKD4PtQ-hhT1kbPSTGRdu8I8NG1uvzcWr4/edit#gid=218846379&fvid=1984765872",
        # "https://docs.google.com/spreadsheets/d/1j_iM9uf_Ao4qgyXZ7-_3BcNnMiY58PrS-Qm57Mkl08g/edit#gid=213858287"
        "https://docs.google.com/spreadsheets/d/105rQwsTJe8VQmZayIktBKJcdPIGMYSCQ4o7J42Tvew0/edit#gid=1166950458"
    ]
    sheet_name_ = SheetNames.MP3_SHEET_NAME
    page_type_ = PageType.NewClassic
    pre_valid = "2021-06-11"

    control_flow = ControlFlow(sheet_name=sheet_name_, urls=urls, page_type=page_type_)
    # ControlFlow_C11
    # control_flow = ControlFlow(sheet_name=sheet_name_, urls=urls, page_type=page_type_, pre_valid=pre_valid)

    # Contribution: pre_valid
    # control_flow.pre_valid_()

    # check_box:
    control_flow.check_box()

    # observe:
    # k = control_flow.observe()
    # print(k)

    # similarity:
    # control_flow.similarity()

    # crawl:
    # control_flow.crawl()

    # checking
    # control_flow.checking()

    # update d9
    # control_flow.update_d9()

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
