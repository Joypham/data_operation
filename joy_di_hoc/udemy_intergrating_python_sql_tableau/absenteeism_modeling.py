import time
from joy_di_hoc.data_scaling import *
from joy_di_hoc.data_cleansing import re_format_file, read_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


class data_modeling():
    def __init__(self, data_file_name):
        # read the 'model' and 'scaler' files which were saved
        with open(data_file_name, 'rb') as raw_data:
            self.raw_data = read_data(path_name=data_file_name)
            self.scaler = None
            self.preprocessed_data = None
            self.data_scaling = None
            self.x_train = None
            self.x_test = None
            self.y_train = None
            self.y_test = None
            self.reg = None
            self.features = None
            self.targets = None

    def load_and_clean_data(self):
        # Step 3: process_file
        df = re_format_file(df=self.raw_data)
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
        absenteeism_scaler = Standardization(columns=scale_columns, copy=True, with_mean=True, with_std=True)
        absenteeism_scaler.fit(X=self.features, y=None)
        self.data_scaling = absenteeism_scaler.transform(self.features)

    # step 6: Train/test split
    def train_test_split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.data_scaling, self.targets, train_size=0.8, shuffle=True,
                                                            random_state=20)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.reg = LogisticRegression().fit(X=x_train, y=y_train)

    def summary_table(self):
    # Extract intercept and coefficients
        intercept = self.reg.intercept_
        coef = self.reg.coef_[0]
        feature_name = self.features.columns.values

        summary_table = pd.DataFrame(
            {
                'column_name': np.append(['intercept'], feature_name),
                'coefficient': np.append(intercept, coef)
            }
        )
        summary_table['odds_ratio'] = np.exp(summary_table.coefficient)
        summary_table = summary_table.sort_values('odds_ratio', ascending=False)
        print(summary_table)

    def predicted_probability(self, x: list):
        pred = self.reg.predict_proba(x)[:, 1]
        return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self, x: list):
        pred_outputs = self.reg.predict(x)
        return pred_outputs

    def save_model(self):
        with open('absenteeism_model', 'wb') as file:
            pickle.dump(self.reg, file)
        with open('absenteeism_scaler', 'wb') as file:
            pickle.dump(self.data_scaling, file)


if __name__ == "__main__":
    start_time = time.time()
    pd.set_option("display.max_rows", None, "display.max_columns", 30, 'display.width', 500)
    data_file_name = '/Users/phamhanh/Documents/Joy đi học/udemy_python_sql_tableau/dataset/Absenteeism_data.csv'
    new_data_file_name = '/Users/phamhanh/PycharmProjects/data_operation/joy_di_hoc/udemy_intergrating_python_sql_tableau/Absenteeism_new_data.csv'
    columns_to_remove = ['id', 'absenteeism_time_in_hours', 'date', 'reason_for_absence']
    data_model = data_modeling(data_file_name=data_file_name)
    data_model.load_and_clean_data()
    data_model.train_test_split()
    # data_model.summary_table()
    # predicted_prob = data_model.predicted_probability(x=data_model.x_train)
    predicted_output_category = data_model.predicted_output_category(x=data_model.x_train)
    data_model.save_model()
    print("\n --- total time to process %s seconds ---" % (time.time() - start_time))
