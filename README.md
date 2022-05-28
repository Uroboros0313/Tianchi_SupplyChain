# 阿里天池天基供应链大赛

## Introduction
阿里天池天基供应链大赛，对公司虚拟资源占用进行销量预测，并指定补货计划

## Dependencies

- python==3.7

- packages

```
    chinese_calendar==1.5.1
    gluonts==0.8.1
    joblib==1.1.0
    lightgbm==2.3.0
    mxnet==1.7.0.post2
    numpy==1.16.6
    pandas==1.1.5
    statsmodels==0.12.2
    tqdm==4.62.2
```

# Usage


```
conda activate your_env
pip install -r requirements.txt
cd ./code
python main.py
```

# File structure

├─code                   
├─data                    
│  ├─Dataset                 
│  ├─forecast                 
│  └─submit                
└─submit                            

# Solution

## Timeseries Forecasting

- Ensemble Learning

由于是预测问题，因此采用了Bagging，及每个模型分别学习，最后进行简单平均的做法。

1. lightgbm
2. deepar
3. holtwinters

其中deepar与lightgbm采用分位数预测的方法，deepar通过修改LSTM的损失函数与蒙特卡洛采样预测输出的分布来进行连续的概率预测，而lightgbm进行分位数回归

每个模型通过计算z值与历史需求标准差计算是否需要补货

由于最后采用的策略是考虑未来三周的预测需求计算再订货点，因此预测一共21天的需求。

| 预测结果           | Z  |
| ------------------ | ---- |
| lightgbm(0.8) | 1.0  |
| deepar(0.8)   | 1.0  |
| deepar             | 2.5  |
| holtwinters        | 2.5  |


## Replenishment

再订货点计算公式：再订货点=预测的未来三周需求+z\*历史需求标准差\*根号提前期，补货量为：max(再订货点-在途库存-库存水平,0)。

# Improvement

比赛途中考虑了几种策略，要么限于当时忙于课题组的项目以及当时代码功底仍然不够好而没有采用，要么做出来效果不好。

1. 考虑时间序列的相关关系，尝试DTW/FastDTW计算时间序列的相关性解决。当然除此之外，根据DeepAR的论文，DeepAR是有记忆不同时间序列的相似Pattern而增强泛化能力的效果的，但DeepAR的training不是我在做，这块具体也没有去看，可以将所有的时间序列输入单一DeepAR模型解决。

2. 关注补货策略，分产品计算补货点，对于数据较为平稳的产品，需求的预测较为精准应该可以提升效果。

3. 时间序列应用在线学习/增量学习机制，由于给出的数据当中，存在一整段的数据为0，可能是虚拟资源下线等情况，可以采用在线学习机制进行识别。项目中提供了0.9分位回归的LGB的增量学习代码，但是由于没有设置early stopping或等等原因，结果并没有提升。




