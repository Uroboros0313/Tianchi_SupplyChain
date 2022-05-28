import pandas as pd
from tqdm import tqdm
from chinese_calendar import is_holiday
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor

class ForecastLGB():
    def __init__(self):
        self.train = pd.read_csv('../data/Dataset/demand_train_B.csv')
        self.test = pd.read_csv('../data/Dataset/demand_test_B.csv')
        self.geo_topo = pd.read_csv('../data/Dataset/geo_topo.csv')
        self.product_topo = pd.read_csv('../data/Dataset/product_topo.csv')
        self.data = None
        self.forecast_path = '../data/forecast/lgb80_result.csv'
        self.lgb_params = {
        'alpha': 0.8,
        'objective': 'quantile',
        'metric': 'quantile',
        'num_leaves': 2**7-1,
        'reg_lambda': 50,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'subsample_freq': 4,
        'learning_rate': 0.015,
        'n_estimators':5000,
        'seed': 1024,
        'n_jobs':-1,
        'silent': True,
        'verbose': -1,
        }

    def makelag(self,data, values, shift):
        lags = [i + shift for i in range(60)]
        rollings = [i for i in range(2, 60)]
        for lag in lags:
            data[f'lag_{lag}'] = values.shift(lag)
        for rolling in rollings:
            data[f's_{shift}_roll_{rolling}_min'] = values.shift(shift).rolling(window=rolling).min()
            data[f's_{shift}_roll_{rolling}_max'] = values.shift(shift).rolling(window=rolling).max()
            data[f's_{shift}_roll_{rolling}_median'] = values.shift(shift).rolling(window=rolling).median()
            data[f's_{shift}_roll_{rolling}_std'] = values.shift(shift).rolling(window=rolling).std()
            data[f's_{shift}_roll_{rolling}_mean'] = values.shift(shift).rolling(window=rolling).mean()
        return data

    def feature_extract(self):
        self.data = pd.concat([self.train, self.test])
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.data = self.data.drop(columns=['Unnamed: 0', 'geography_level', 'product_level'])
        self.data.columns = ['unit', 'ts', 'qty', 'geography_level_3', 'product_level_2']
        self.data = pd.merge(self.data, self.geo_topo, on=['geography_level_3'])
        self.data = pd.merge(self.data, self.product_topo, on=['product_level_2'])
        self.data = self.data.sort_values(
            ['product_level_1', 'product_level_2', 'geography_level_1', 'geography_level_2', 'geography_level_3',
             'unit', 'ts'])
        self.data['qty'] = self.data.groupby('unit')['qty'].diff()
        self.data = self.data.dropna()



        for date in pd.date_range('2021-06-08', '2021-06-28'):
            append_data = pd.DataFrame(columns=self.data.columns)
            append_data[
                ['product_level_1', 'product_level_2', 'geography_level_1', 'geography_level_2', 'geography_level_3',
                 'unit']] = self.data[
                ['product_level_1', 'product_level_2', 'geography_level_1', 'geography_level_2', 'geography_level_3',
                 'unit']].drop_duplicates()
            append_data['ts'] = date
            self.data = pd.concat([self.data, append_data], ignore_index=True)
        self.data['qty'] = self.data['qty'].astype('float')
        self.data = self.data.sort_values(
            ['product_level_1', 'product_level_2', 'geography_level_1', 'geography_level_2', 'geography_level_3',
             'unit', 'ts'])

        self.data = self.data.groupby(['unit']).apply(lambda x: self.makelag(x, x['qty'], 21))

        for func in ['mean', 'std']:
            for col in ['unit', 'geography_level_3', 'geography_level_2', 'geography_level_1', 'product_level_2',
                        'product_level_1']:
                tmp_ss = self.data[self.data['ts'] <= '2021-03-01'].groupby(col)['qty'].agg(func)
                tmp_ss.name = f'{func}_{col}'
                self.data = pd.merge(self.data, tmp_ss, on=col)

        for col in ['geography_level_3', 'geography_level_2', 'geography_level_1', 'product_level_2', 'product_level_1',
                    'unit']:
            unique_value = self.data[col].unique()
            map_value = [i for i in range(len(unique_value))]
            map_dict = dict(zip(unique_value, map_value))
            self.data[col] = self.data[col].map(map_dict)


        self.data['quarter'] = self.data['ts'].apply(lambda x: x.quarter).astype('category')
        self.data['month'] = self.data['ts'].apply(lambda x: x.month).astype('category')
        self.data['weekday'] = self.data['ts'].apply(lambda x: x.weekday()).astype('category')
        self.data['week'] = self.data['ts'].apply(lambda x: x.week).astype('category')
        self.data['day'] = self.data['ts'].apply(lambda x: x.day).astype('category')
        self.data['is_weekend'] = self.data['ts'].apply(lambda x: 1 if x in [5, 6] else 0)
        self.data['is_holiday'] = self.data['ts'].apply(is_holiday)

        return map_dict


    def forecast(self,map_dict):
        inverse_map_dict = dict([val, key] for key, val in map_dict.items())
        pred_frame_list = []
        for date in tqdm(pd.date_range(start='2021-03-08', end='2021-06-07', freq='7d')):
            end_date = date + pd.DateOffset(21)
            val_date = date - pd.DateOffset(21)

            train = self.data[self.data['ts'] <= val_date]
            val = self.data[(self.data['ts'] > val_date) & (self.data['ts'] <= date)]
            test = self.data[(self.data['ts'] > date) & (self.data['ts'] <= end_date)]
            X_train = train.drop(columns=['qty', 'ts'])
            y_train = train['qty']
            X_val = val.drop(columns=['qty', 'ts'])
            y_val = val['qty']
            X_test = test.drop(columns=['qty', 'ts'])
            y_test = test['qty']

            model = LGBMRegressor(**self.lgb_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100)

            pred = model.predict(X_test)
            pred_frame = test[['unit', 'ts']]
            pred_frame['pred'] = pred
            pred_frame['unit'] = pred_frame['unit'].map(inverse_map_dict)
            pred_frame = pred_frame.pivot(index='ts', columns='unit', values='pred').reset_index()
            pred_frame.columns.name = None
            pred_frame['decision_day'] = date
            pred_frame_list.append(pred_frame)

        final_result = pd.concat(pred_frame_list)
        final_result.to_csv(self.forecast_path)

    def run(self):
        map_dict = self.feature_extract()
        self.forecast(map_dict)

if __name__ == "__main__":
    forecast_utils = ForecastLGB()
    forecast_utils.run()









