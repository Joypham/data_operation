from google_spreadsheet_api.function import get_df_from_speadsheet, get_list_of_sheet_title, update_value, \
    creat_new_sheet_and_update_data_from_df, get_gsheet_name
from Data_lake_process.class_definition import get_key_value_from_gsheet_info, add_key_value_from_gsheet_info, get_gsheet_id_from_url
from datetime import datetime


def update_data_reports(gsheet_info: object, status: str = None, count_complete: int = 0, count_incomlete: int = 0,
                        notice: str = None):
    gsheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_name')
    sheet_name = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='sheet_name')
    gsheet_id = get_key_value_from_gsheet_info(gsheet_info=gsheet_info, key='gsheet_id')
    gsheet_url = f"https://docs.google.com/spreadsheets/d/{gsheet_id}"
    # https://docs.google.com/spreadsheets/d/1MHDksbs-RKXhZZ-LRgRhVy_ldAxK8lSzyoJK4sA_Uyo
    print(f"updating data_reports gsheet_name: {gsheet_name}, type: {sheet_name}")
    reports_df = get_df_from_speadsheet(gsheet_id="1MHDksbs-RKXhZZ-LRgRhVy_ldAxK8lSzyoJK4sA_Uyo", sheet_name="demo")
    row_index = reports_df.index
    for i in row_index:
        reports_gsheet_name = reports_df['gsheet_name'].loc[i]
        reports_sheet_name = reports_df['type'].loc[i]
        if gsheet_name == reports_gsheet_name and sheet_name == reports_sheet_name:
            range_to_update = f"demo!A{i + 2}"
            break
        else:
            range_to_update = f"demo!A{row_index.n_estimators_stop + 2}"
    list_result = [
        [gsheet_name, gsheet_url, sheet_name, f"{datetime.now()}", status, count_complete, count_incomlete, notice]]
    update_value(list_result=list_result, grid_range_to_update=range_to_update,
                 gsheet_id="1MHDksbs-RKXhZZ-LRgRhVy_ldAxK8lSzyoJK4sA_Uyo")