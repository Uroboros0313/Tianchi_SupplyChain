import pandas as pd
import numpy as np
from tqdm import tqdm

class ReplenishUnit:
    def __init__(self,
                 unit,
                 demand_hist,
                 intransit,
                 qty_replenish,
                 qty_inventory_today,
                 qty_using_today,
                 arrival_sum,
                 lead_time,
                 demand_forecast,
                 z
                 ):
        '''
        记录各补货单元状态
        :param unit:
        :param demand_hist: 净需求历史
        :param intransit: 补货在途
        :param qty_replenish: 补货记录
        :param qty_inventory_today: 当前可用库存
        :param qty_using_today: 当前已用库存（使用量）
        :param arrival_sum: 补货累计到达
        :param lead_time: 补货时长，交货时间
        '''
        self.unit = unit
        self.demand_hist = demand_hist
        self.demand_forecast = demand_forecast
        self.intransit = intransit
        self.qty_replenish = qty_replenish
        self.qty_inventory_today = qty_inventory_today
        self.qty_using_today = qty_using_today
        self.arrival_sum = arrival_sum
        self.lead_time = lead_time
        self.record_hist = pd.DataFrame(columns=['ts', 'qty_inventory', 'qty_using', 'stock_out', 'demand'])
        self.z=z

    def update(self,
               date,
               arrival_today,
               demand_today):
        '''
        每日根据当天补货到达与当日净需求更新状态
        :param date:
        :param arrival_today: 当天补货到达
        :param demand_today: 当天净需求
        :return:
        '''
        self.qty_inventory_today += arrival_today
        self.arrival_sum += arrival_today
        inv_today = self.qty_inventory_today
        stock_out = max((demand_today - self.qty_inventory_today), 0)
        if demand_today < 0:
            self.qty_inventory_today = self.qty_inventory_today + min(-demand_today, self.qty_using_today)
        else:
            self.qty_inventory_today = max(self.qty_inventory_today - demand_today, 0.0)
        self.qty_using_today = max(self.qty_using_today + min(demand_today, inv_today), 0.0)
        self.demand_hist = self.demand_hist.append({"ts": date, "unit": self.unit, "qty": demand_today},
                                                   ignore_index=True)
        self.record_hist = self.record_hist.append({'ts': date, 'qty_inventory': self.qty_inventory_today,
                                                    'qty_using': self.qty_using_today, 'stock_out': stock_out,
                                                    'demand': demand_today}, ignore_index=True)

    def forecast_function(self,
                          date):

        # demand_average = np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])
        # return [demand_average] * 90
        # 简单指数平滑法预测
        # model = ExponentialSmoothing(self.demand_hist["qty"].values).fit()
        # return model.forecast(21)
        return self.demand_forecast[self.demand_forecast['decision_day']==date][self.unit].values

    def replenish_function(self,
                           date):
        '''
        根据当前状态判断需要多少的补货量
        补货的策略由选手决定，这里只给一个思路
        :param date:
        :return:
        '''
        replenish = 0.0
        if date.dayofweek != 0:
            # 周一为补货决策日，非周一不做决策
            pass
        else:
            # 预测未来需求量
            qty_demand_forecast = self.forecast_function(date)

            # 计算在途的补货量
            qty_intransit = sum(self.intransit) - self.arrival_sum

            # 安全库存 用来抵御需求的波动性 选手可以换成自己的策略
            # safety_stock = (max(self.demand_hist["qty"].values[-3 * self.lead_time:]) - (np.mean(self.demand_hist["qty"].values[- 3 * self.lead_time:]))) * self.lead_time
            safety_stock = self.z * np.std(self.demand_hist["qty"].values) * np.sqrt(self.lead_time)
            # safety_stock = 0 * np.std(self.demand_hist["qty"].values) * np.sqrt(self.lead_time)
            # # 再补货点，用来判断是否需要补货 选手可以换成自己的策略
            reorder_point = sum(qty_demand_forecast) + safety_stock
            #
            # # 判断是否需要补货并计算补货量，选手可以换成自己的策略，可以参考赛题给的相关链接
            if self.qty_inventory_today + qty_intransit < reorder_point:
                replenish = reorder_point - (self.qty_inventory_today + qty_intransit)

            self.qty_replenish.at[date] = replenish
            self.intransit.at[date + self.lead_time * date.freq] = replenish


class SupplyChain:
    def __init__(self, z ,forecast_path):
        self.using_hist = pd.read_csv("../data/Dataset/demand_train_B.csv")
        self.using_future = pd.read_csv("../data/Dataset/demand_test_B.csv")
        self.inventory = pd.read_csv("../data/Dataset/inventory_info_B.csv")
        self.weight = pd.read_csv('../data/Dataset/weight_B.csv')
        self.forecast_path=forecast_path
        self.forecast=pd.read_csv(forecast_path)
        self.z=z
        self.last_dt = pd.to_datetime("20210301")
        self.start_dt = pd.to_datetime("20210302")
        self.end_dt = pd.to_datetime("20210607")
        self.lead_time = 14

    def run(self):
        self.forecast['ts']=self.forecast['ts'].apply(lambda x: pd.to_datetime(x))
        self.forecast['decision_day'] = self.forecast['decision_day'].apply(lambda x: pd.to_datetime(x))
        self.using_hist["ts"] = self.using_hist["ts"].apply(lambda x: pd.to_datetime(x))
        self.using_future["ts"] = self.using_future["ts"].apply(lambda x: pd.to_datetime(x))
        qty_using = pd.concat([self.using_hist, self.using_future])
        date_list = pd.date_range(start=self.start_dt, end=self.end_dt)
        unit_list = self.using_future["unit"].unique()
        res = pd.DataFrame(columns=["unit", "ts", "qty"])

        replenishUnit_dict = {}
        demand_dict = {}

        # 初始化，记录各补货单元在评估开始前的状态
        for chunk in qty_using.groupby("unit"):
            unit = chunk[0]
            demand = chunk[1]
            demand.sort_values("ts", inplace=True, ascending=True)

            # 计算净需求量
            demand["diff"] = demand["qty"].diff().values
            demand["qty"] = demand["diff"]
            del demand["diff"]
            demand = demand[1:]
            replenishUnit_dict[unit] = ReplenishUnit(unit=unit,
                                                     demand_hist=demand[demand["ts"] < self.start_dt],
                                                     intransit=pd.Series(index=date_list.tolist(),
                                                                         data=[0.0] * (len(date_list))),
                                                     qty_replenish=pd.Series(index=date_list.tolist(),
                                                                             data=[0.0] * (len(date_list))),
                                                     qty_inventory_today=
                                                     self.inventory[self.inventory["unit"] == unit]["qty"].values[0],
                                                     qty_using_today=self.using_hist[
                                                         (self.using_hist["ts"] == self.last_dt) & (
                                                                     self.using_hist["unit"] == unit)]["qty"].values[0],
                                                     arrival_sum=0.0,
                                                     lead_time=self.lead_time,
                                                     demand_forecast=self.forecast[['ts','decision_day',unit]],
                                                     z=self.z)
            # 记录评估周期内的净需求量
            demand_dict[unit] = demand[(demand["unit"] == unit) & (demand["ts"] >= self.start_dt)]
        print('判断补货量中...')
        for date in tqdm(date_list):
            # 按每日净需求与每日补货到达更新状态，并判断补货量
            for unit in unit_list:
                demand = demand_dict[unit]
                demand_today = demand[demand["ts"] == date]["qty"].values[0]
                arrival = replenishUnit_dict[unit].intransit.get(date, default=0.0)
                replenishUnit_dict[unit].update(date=date,
                                                arrival_today=arrival,
                                                demand_today=demand_today)
                replenishUnit_dict[unit].replenish_function(date)

        for unit in unit_list:
            res_unit = replenishUnit_dict[unit].qty_replenish
            res_unit = pd.DataFrame({"unit": unit,
                                     "ts": res_unit.index,
                                     "qty": res_unit.values})
            res_unit = res_unit[res_unit["ts"].apply(lambda x: x.dayofweek == 0)]
            res = pd.concat([res, res_unit])

        # 评估函数
        score_df = pd.DataFrame(columns=['unit', 'inv_rate', 'f_sla', 'weight'])
        for unit in unit_list:
            w = self.weight[self.weight['unit'] == unit]['weight'].values[0]
            record_hist = replenishUnit_dict[unit].record_hist
            inv_rate = np.mean(record_hist['qty_inventory'] / (record_hist['qty_inventory'] + record_hist['qty_using']))
            try:
                sla = 1 - np.sum(record_hist['stock_out']) / self.maxSubArray(list(record_hist['demand']))
            except:
                sla = 1
            f_sla = 1 / (1 + np.exp(-10 * (sla - 0.5)))
            score_df = score_df.append({'unit': unit, 'inv_rate': inv_rate,
                                        'f_sla': f_sla, 'weight': w}, ignore_index=True)
        inv_rate_sum = np.sum((1 - score_df['inv_rate']) * score_df['weight'])
        sla_sum = np.sum(score_df['f_sla'] * score_df['weight'])
        final_score = 0.5 * inv_rate_sum + 0.5 * sla_sum
        print(self.z)
        print(f"库存分数：{inv_rate_sum}")
        print(f"缺货率：{sla_sum}")
        print(f'综合指标：{final_score}')
        # 输出结果
        res.to_csv(f"../data/submit/{self.forecast_path.split('/')[-1]}")

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # onesum维护当前的和
        onesum = 0
        maxsum = nums[0]
        for i in range(len(nums)):
            onesum += nums[i]
            maxsum = max(maxsum, onesum)
            # 出现onesum<0的情况，就设为0，重新累积和
            if onesum < 0:
                onesum = 0
        return maxsum


if __name__ == '__main__':
    supply_chain = SupplyChain(1,'../data/forecast/deepar_result_80.csv')
    supply_chain.run()
