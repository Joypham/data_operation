import pandas as pd
import time
import re
from joy_di_hoc.udemy_intergrating_python_sql_tableau.data_scaling import *
from joy_di_hoc.udemy_intergrating_python_sql_tableau.data_cleansing import re_format_file, read_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle


class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files which were saved
        with open(model_file, 'rb') as model_file, open(scaler_file, 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            self.targets = None
            self.original_df = None
            self.features = None
            self.preprocessed_data = None

    def load_and_clean_data(self, data_file):
        self.original_df = read_data(path_name=data_file)
        # Step 3: process_file
        df = re_format_file(df=self.original_df)
        df['date'] = pd.to_datetime(df['date'])
        df['month_value'] = df['date'].dt.month
        df['day_of_the_week'] = df['date'].dt.dayofweek
        df['education'] = df['education'].map({1: 0, 2: 1, 3: 1, 4: 1})
        # break moderately absence and excessively absence by median
        breaking_point = df['absenteeism_time_in_hours'].median()

        # Step 4: Determine target variable and elimination
        reason_for_absence = pd.get_dummies(df['reason_for_absence'], drop_first=True)
        reason1_column = pd.DataFrame({'reason 1': reason_for_absence.loc[:, 1:14].max(axis=1)})
        reason2_column = pd.DataFrame({'reason 2': reason_for_absence.loc[:, 15:17].max(axis=1)})
        reason3_column = pd.DataFrame({'reason 3': reason_for_absence.loc[:, 18:21].max(axis=1)})
        reason4_column = pd.DataFrame({'reason 4': reason_for_absence.loc[:, 22:].max(axis=1)})
        df = pd.concat([df, reason1_column, reason2_column, reason3_column, reason4_column], axis=1)
        self.targets = df['absenteeism_time_in_hours'].apply(
            lambda x: 0 if x <= breaking_point else 1)
        self.features = df.drop(columns_to_remove, axis=1)
        self.preprocessed_data = self.features.join(self.targets)

        # Step 5: data scaling
        scale_columns = [x for x in self.features.columns.values if x not in columns_to_remove]
        absenteeism_scaler = CustomScaler(columns=scale_columns, copy=True, with_mean=True, with_std=True)
        absenteeism_scaler.fit(X=self.features, y=None)
        self.data = absenteeism_scaler.transform(self.features)

    def predicted_probability(self):
        if self.data is not None:
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    data_file_name = '/Users/phamhanh/Documents/Joy đi học/udemy_python_sql_tableau/dataset/Absenteeism_data.csv'
    new_data_file_name = '/Users/phamhanh/PycharmProjects/data_operation/joy_di_hoc/udemy_intergrating_python_sql_tableau/Absenteeism_new_data.csv'
    columns_to_remove = ['id', 'absenteeism_time_in_hours', 'date', 'reason_for_absence']

    model = absenteeism_model(
        model_file='/Users/phamhanh/PycharmProjects/data_operation/joy_di_hoc/udemy_intergrating_python_sql_tableau/absenteeism_model',
        scaler_file='/Users/phamhanh/PycharmProjects/data_operation/joy_di_hoc/udemy_intergrating_python_sql_tableau/absenteeism_scaler')
    model.load_and_clean_data(data_file=data_file_name)
    k = model.predicted_outputs()

    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
