import pandas as pd
def get_df_from_query(query):
    column_name = query.column_descriptions
    list_column_name = []
    for column in column_name:
        list_column_name.append(column.get('name'))
    df = pd.DataFrame(query.all(), columns=list_column_name)
    return df
