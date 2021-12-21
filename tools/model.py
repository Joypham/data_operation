import time
import pandas as pd
from google_spreadsheet_api.function import get_df_from_speadsheet, get_gsheet_name, update_value

from Data_process.crawlingtask import crawl_itunes_album
from core import query_path

if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 50, 'display.width', 1000)
    # with open(query_path, "a+") as f:
    #     df = get_df_from_speadsheet(gsheet_id="1HUnal5ZfTngeSlKVCLH0kTnz0ZMa_RwvDv5uTqJSM6Q", sheet_name="joy")
        # gsheet_name = get_gsheet_name(gsheet_id="1HUnal5ZfTngeSlKVCLH0kTnz0ZMa_RwvDv5uTqJSM6Q")
        # sheet_name = "joy"
    #     pic = f"{gsheet_name}_{sheet_name}"
    #     row_index = df.index
    #     for i in row_index:
    #         itune_id = df.id.loc[i]
    #         region = df["region"].loc[i]
    #         crawling_task = crawl_itunes_album(ituneid=itune_id, pic=pic, region=region)
    #         print(crawling_task)
    #         f.write(crawling_task)

    df = get_df_from_speadsheet(gsheet_id="1J0tfInOX5VFnC0QM2CVnPb2i4YblF1EVFoZAOtEslHo", sheet_name="MP_3")
    print(df.head(10))
    print("--- %s seconds ---" % (time.time() - start_time))