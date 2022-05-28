import pandas as pd
import numpy as np
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import mxnet as mx
mx.random.seed(0)
np.random.seed(0)

class ForecastDeepAR:
    def __init__(self):
        self.train = pd.read_csv('../data/Dataset/demand_train_B.csv')
        self.test = pd.read_csv('../data/Dataset/demand_test_B.csv')
        self.geo_topo = pd.read_csv('../data/Dataset/geo_topo.csv')
        self.product_topo = pd.read_csv('../data/Dataset/product_topo.csv')

    def run(self):
        data = pd.concat([self.train, self.test])
        data['ts'] = pd.to_datetime(data['ts'])
        data = data.drop(columns=['Unnamed: 0', 'geography_level', 'product_level'])
        data.columns = ['unit', 'ts', 'qty', 'geography_level_3', 'product_level_2']
        data = pd.merge(data, self.geo_topo, on=['geography_level_3'])
        data = pd.merge(data, self.product_topo, on=['product_level_2'])
        data = data.sort_values(
            ['product_level_1', 'product_level_2', 'geography_level_1', 'geography_level_2', 'geography_level_3',
             'unit', 'ts'])
        data['qty'] = data.groupby('unit')['qty'].diff()
        data = data.dropna()
        target_data = data.pivot(
            index=['product_level_1', 'product_level_2', 'geography_level_1', 'geography_level_2', 'geography_level_3',
                   'unit'], columns='ts', values='qty')
        cat_data = target_data.index.to_frame().reset_index(drop=True)
        for date in pd.date_range('2021-06-08', '2021-06-28'):
            target_data[date] = np.nan
        unit = cat_data["unit"].astype('category').cat.codes.values
        state_ids_un, state_ids_counts = np.unique(unit, return_counts=True)
        product_level_1 = cat_data["product_level_1"].astype('category').cat.codes.values
        product_level_1_un, product_level_1_counts = np.unique(product_level_1, return_counts=True)
        product_level_2 = cat_data["product_level_2"].astype('category').cat.codes.values
        product_level_2_un, product_level_2_counts = np.unique(product_level_2, return_counts=True)
        geography_level_1 = cat_data["geography_level_1"].astype('category').cat.codes.values
        geography_level_1_un, geography_level_1_counts = np.unique(geography_level_1, return_counts=True)
        geography_level_2 = cat_data["geography_level_2"].astype('category').cat.codes.values
        geography_level_2_un, geography_level_2_counts = np.unique(geography_level_2, return_counts=True)
        geography_level_3 = cat_data["geography_level_3"].astype('category').cat.codes.values
        geography_level_3_un, geography_level_3_counts = np.unique(geography_level_3, return_counts=True)
        stat_cat_list = [unit, product_level_1, product_level_2, geography_level_1, geography_level_2,
                         geography_level_3]

        stat_cat = np.concatenate(stat_cat_list)
        stat_cat = stat_cat.reshape(len(stat_cat_list), len(unit)).T

        stat_cat_cardinalities = [len(state_ids_un), len(product_level_1_un), len(product_level_2_un), len(geography_level_1_un),
                                  len(geography_level_2_un), len(geography_level_3_un), ]
        dates = [pd.Timestamp("2018-06-05", freq='1D') for _ in range(len(target_data))]
        forecast_dict = {}

        for date in tqdm(pd.date_range(start='2021-03-08', end='2021-06-07', freq='7d')):
            tmp_target_data = target_data.loc[:, :date + pd.DateOffset(21)]
            test_target_values = tmp_target_data.values
            train_target_values = tmp_target_data.values[:, :-21]
            train_ds = ListDataset([
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: fsc
                }
                for (target, start, fsc) in zip(train_target_values,
                                                dates,
                                                stat_cat)
            ], freq="D")

            test_ds = ListDataset([
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: fsc
                }
                for (target, start, fsc) in zip(test_target_values,
                                                dates,
                                                stat_cat)
            ], freq="D")
            estimator = DeepAREstimator(
                context_length=35,
                prediction_length=21,
                num_layers=2,
                num_cells=40,
                freq="D",
                use_feat_static_cat=True,
                cardinality=stat_cat_cardinalities,
                trainer=Trainer(
                    learning_rate=1e-3,
                    num_batches_per_epoch=100,
                    epochs=20,
                )
            )

            predictor = estimator.train(train_ds)
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=test_ds,
                predictor=predictor,
                num_samples=100
            )
            forecasts = list(tqdm(forecast_it, total=len(test_ds)))
            forecast_dict[date] = forecasts

        result_list = []
        for date in tqdm(pd.date_range(start='2021-03-08', end='2021-06-07', freq='7d')):
            result = pd.DataFrame(columns=cat_data['unit'])
            for i in range(len(result.columns)):
                result.iloc[:, i] = forecast_dict[date][i].quantile(0.8)
            result['ts'] = pd.date_range(date + pd.DateOffset(1), date + pd.DateOffset(21))
            result['decision_day'] = date
            result_list.append(result)
        final_result = pd.concat(result_list)
        final_result.to_csv('../data/forecast/deepar_result_80.csv', index=False)

        result_list = []
        for date in tqdm(pd.date_range(start='2021-03-08', end='2021-06-07', freq='7d')):
            result = pd.DataFrame(columns=cat_data['unit'])
            for i in range(len(result.columns)):
                result.iloc[:, i] = forecast_dict[date][i].mean
            result['ts'] = pd.date_range(date + pd.DateOffset(1), date + pd.DateOffset(21))
            result['decision_day'] = date
            result_list.append(result)
        final_result = pd.concat(result_list)
        final_result.to_csv('../data/forecast/deepar_result_mean.csv', index=False)

if __name__ == '__main__':
    deepar=ForecastDeepAR()
    deepar.run()