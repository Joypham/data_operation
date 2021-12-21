import time
import pandas as pd
from Data_process.crawlingtask import sheet_type
from Data_process.data_lake_standard import process_image


class page_types:
    album_page = {}
    single_page = {}


def process_page():
    image_df = process_image(urls=urls, sheet_info=sheet_info, sheet_name_core=sheet_name_core)
    print(image_df)


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)

    urls = [
        'https://docs.google.com/spreadsheets/d/1mBJcQvqobNfISSrandRrBCpQCY8vocDi7w6HPUXOY-A/edit#gid=0',
        'https://docs.google.com/spreadsheets/d/16Y4E748cOxHrewOEQ-ecQqcz9j1phlHLl-PgX0SG3vo/edit#gid=0'
    ]
    sheet_name_core = '08.03.2021'
    sheet_info = sheet_type.ARTIST_IMAGE
    process_page()


    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
