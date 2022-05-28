import os
from forecast_deepar import ForecastDeepAR
from forecast_holtwinters import ForecastHoltWinters
from forecast_lgb import ForecastLGB
from supply_chain import SupplyChain
import pandas as pd

def main():
    if not os.path.exists('../data/forecast'):
        os.mkdir('../data/forecast')
    if not os.path.exists('../data/submit'):
        os.mkdir('../data/submit')
    holt_winters=ForecastHoltWinters()
    holt_winters.run()
    lgb=ForecastLGB()
    lgb.run()
    deepar=ForecastDeepAR()
    deepar.run()
    supply_chain1 = SupplyChain(2.5, '../data/forecast/holtwinters_result.csv')
    supply_chain1.run()
    supply_chain2 = SupplyChain(1, '../data/forecast/lgb80_result.csv')
    supply_chain2.run()
    supply_chain3 = SupplyChain(2.5, '../data/forecast/deepar_result_mean.csv')
    supply_chain3.run()
    supply_chain4 = SupplyChain(1, '../data/forecast/deepar_result_80.csv')
    supply_chain4.run()
    result_df_list = []
    for result in os.listdir('../data/submit'):
        result_df = pd.read_csv(f'../data/submit/{result}')
        result_df_list.append(result_df)
    merge_result = result_df.copy()[['unit', 'ts', 'qty']]
    merge_result['qty'] = 0
    for result_df in result_df_list:
        merge_result['qty'] += result_df['qty']
    merge_result['qty'] = merge_result['qty'] / len(result_df_list)
    merge_result.to_csv('../submit/submit.csv')

if __name__ == '__main__':
    main()