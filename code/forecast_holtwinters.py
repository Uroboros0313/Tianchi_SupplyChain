import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


class ForecastHoltWinters():
    def __init__(self):
        self.train = pd.read_csv('../data/Dataset/demand_train_B.csv')
        self.test = pd.read_csv('../data/Dataset/demand_test_B.csv')
        self.geo_topo = pd.read_csv('../data/Dataset/geo_topo.csv')
        self.product_topo = pd.read_csv('../data/Dataset/product_topo.csv')
        self.data = None
        self.unit_list = None
        self.forecast_path = '../data/forecast/holtwinters_result.csv'

    def feature_extract(self):
        self.data = pd.concat([self.train, self.test])
        self.data['ts'] = pd.to_datetime(self.data['ts'])
        self.data = self.data[['unit', 'ts', 'qty']]
        self.data = self.data.sort_values(['unit', 'ts'])
        self.data['qty'] = self.data.groupby('unit')['qty'].diff()
        self.data.dropna(inplace=True)
        for date in pd.date_range('2021-06-08', '2021-06-28'):
            append_data = pd.DataFrame(columns=self.data.columns)
            append_data[['unit']] = self.data[['unit']].drop_duplicates()
            append_data['ts'] = date
            self.data = pd.concat([self.data, append_data], ignore_index=True)
            self.data['qty'] = self.data['qty'].astype('float')
            self.data = self.data.sort_values(['unit', 'ts'])

        self.unit_list = list(self.data.unit.unique())

    def forecast(self, date):
        end_date = date + pd.DateOffset(21)
        unit_dict = dict()
        for unit in self.unit_list:
            train = self.data[(self.data['ts'] <= date) & (self.data["unit"] == unit)]
            test = self.data[(self.data['ts'] > date) \
                             & (self.data['ts'] <= end_date) \
                             & (self.data["unit"] == unit)]
            model = ExponentialSmoothing(train["qty"].values).fit()
            pred = model.forecast(21)
            unit_dict[unit] = pred

        unit_dict["ts"] = test.ts.values
        unit_dict["decision_day"] = test.ts.values.shape[0] * [date]

        decision_frame = pd.DataFrame(unit_dict)
        return decision_frame

    def run(self):
        self.feature_extract()
        pred_frame_list = Parallel(n_jobs=-1)(delayed(self.forecast)(date) for date in
                                              pd.date_range(start='2021-03-08', end='2021-06-07', freq='7d'))  # 并行化处理
        pred_frame = pd.concat(pred_frame_list)
        pred_frame.to_csv(self.forecast_path, index=False)


if __name__ == "__main__":
    fc_utils = ForecastHoltWinters()
    fc_utils.run()
