from google_spreadsheet_api.function import get_df_from_speadsheet, creat_new_sheet_and_update_data_from_df, get_gsheet_name
from core.models.crawlingtask_action_master import V4CrawlingTaskActionMaster
from core.crud.sql import artist, album
import pandas as pd
import time
from core import query_path
from colorama import Fore, Style
from Data_lake_process.crawlingtask import crawl_image, crawl_youtube_mp3, crawl_youtube_mp4, crawl_itunes_album
from Data_lake_process.class_definition import WhenExist, PageType, SheetNames, merge_file, DataReports, \
    get_key_value_from_gsheet_info, add_key_value_from_gsheet_info, get_gsheet_id_from_url
from Data_lake_process.new_check_box_standard import youtube_check_box, s11_checkbox, update_s11_check_box
from Data_lake_process.data_report import update_data_reports
from Data_lake_process.checking_accuracy_and_crawler_status import checking_image_youtube_accuracy, automate_checking_status
from crawl_itune.functions import get_itune_id_region_from_itune_url
from core.crud.get_df_from_query import get_df_from_query
from core.crud.sql.query_supporter import get_crawlingtask_info, get_s11_crawlingtask_info


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
        count_incomplete = df_incomplete_to_upload.index.stop
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
                                                object_type=get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='object_type'),
                                                url=x['url_to_add'],
                                                objectid=x['uuid'],
                                                when_exists=when_exists,
                                                pic=f"{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='gsheet_name')}_{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='sheet_name')}",
                                                priority=get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='page_priority')
                                   ),
                                   axis=1)
            query_pandas_to_csv(df=df, column='query')

    def checking_image_crawler_status(self):
        print("checking accuracy")
        df = self.image_filter().copy()
        gsheet_infos = list(set(df.gsheet_info.tolist()))
        # step 1.1: checking accuracy
        checking_accuracy_result = checking_image_youtube_accuracy(df=df, actionid=V4CrawlingTaskActionMaster.ARTIST_ALBUM_IMAGE)
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
        df = self.youtube_filter().copy()
        gsheet_infos = list(set(df.gsheet_info.tolist()))
        # step 1.1: checking accuracy
        checking_accuracy_result = checking_image_youtube_accuracy(df=df, actionid=V4CrawlingTaskActionMaster.DOWNLOAD_VIDEO_YOUTUBE)
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
            automate_checking_status(df=df, actionid=V4CrawlingTaskActionMaster.DOWNLOAD_VIDEO_YOUTUBE)

            # #     # Step 3: upload image cant crawl
            # print(checking_accuracy_result)


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

    def crawl_s11_datalake(self, when_exists: str = WhenExist.REPLACE):
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
                                                      priority=get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='page_priority'),
                                                      is_new_release=is_new_release,
                                                      pic=f"{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='gsheet_name')}_{get_key_value_from_gsheet_info(gsheet_info=x['gsheet_info'], key='sheet_name')}",
                                                      region=x['region']
                                                      ),
                                   axis=1)
        query_pandas_to_csv(df=df, column='query')

    def checking_s11_crawler_status(self):
        print("checking accuracy")
        df = self.s11_filter().copy()
        gsheet_infos = list(set(df.gsheet_info.tolist()))
        for gsheet_info in gsheet_infos:
            gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
            sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
            PIC_taskdetail = f"{gsheet_name}_{sheet_name}"
            k = get_df_from_query(get_s11_crawlingtask_info(pic=PIC_taskdetail))
            print(k)
            # print(PIC_taskdetail)

        # step 1.1: checking accuracy


    #     accuracy_checking = list(set(checking_accuracy_result['check'].tolist()))
    #
    #     if accuracy_checking != [True]:
    #         print(checking_accuracy_result[['uuid', 'check', 'status', 'crawlingtask_id']])
    #         # Step 1.2: update data_reports if checking accuracy fail
    #         for gsheet_info in gsheet_infos:
    #             update_data_reports(gsheet_info=gsheet_info, status=DataReports.status_type_processing,
    #                                 notice="check accuracy fail")
    #     # Step 2: auto checking status
    #     else:
    #         print("checking accuracy correctly, now checking status")
    #         automate_checking_status(df=df, actionid=V4CrawlingTaskActionMaster.ARTIST_ALBUM_IMAGE)
    #         # Step 3: upload image cant crawl
    #         upload_image_cant_crawl(checking_accuracy_result=checking_accuracy_result, sheet_name=self.sheet_name)


class ControlFlow:
    def __init__(self, sheet_name: str, urls: list, page_type: object):
        self.page_type = page_type
        self.urls = urls
        self.sheet_name = sheet_name

    def check_box(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            check_box = image_working.check_box()
        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            check_box = youtube_working.check_box()
        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            check_box = s11_working.check_box()

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

    def crawl(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            image_working.crawl_image_datalake()

        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            youtube_working.crawl_mp3_mp4_youtube_datalake()

        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return s11_working.crawl_s11_datalake()

    def checking(self):
        if self.sheet_name in (SheetNames.ARTIST_IMAGE, SheetNames.ALBUM_IMAGE):
            image_working = ImageWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            image_working.checking_image_crawler_status()

        elif self.sheet_name in (SheetNames.MP3_SHEET_NAME, SheetNames.MP4_SHEET_NAME):
            youtube_working = YoutubeWorking(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            youtube_working.checking_youtube_crawler_status()

        elif self.sheet_name == SheetNames.S_11:
            s11_working = S11Working(sheet_name=self.sheet_name, urls=self.urls, page_type=self.page_type)
            return s11_working.checking_s11_crawler_status()


if __name__ == "__main__":
    start_time = time.time()

    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    with open(query_path, "w") as f:
        f.truncate()
    urls = [
        "https://docs.google.com/spreadsheets/d/14J67QV_u353oYmNk85EIWJ0HU4NnCFThvSBcer992FQ/edit#gid=0"
    ]
    sheet_name_ = SheetNames.S_11
    page_type_ = PageType.NewClassic

    # k = S11Working(sheet_name=sheet_name_, urls=urls, page_type=page_type_)
    # print(k.original_file)

    control_flow = ControlFlow(sheet_name=sheet_name_, urls=urls, page_type=page_type_)
    # check_box:
    # control_flow.check_box()

    # observe:
    k = control_flow.observe()
    print(k)


    # crawl:
    control_flow.crawl()

    # checking
    # control_flow.checking()

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
