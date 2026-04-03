# KHAOS A股 iterA1 Training Report

## Checkpoints

- best_path: `D:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\iterA1_ashare\iterA1_ashare_best.pth`
- final_path: `D:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\02_核心代码\模型权重备份\iterA1_ashare\iterA1_ashare_final.pth`
- best_val_loss: `16.77544933622298`
- best_score: `0.03313436234290007`
- final_env: `{'torch': '2.6.0+cu124', 'cuda': '12.4', 'device': 'cuda'}`

## Overall OOS

- breakout_corr: `0.2830`
- reversion_corr: `0.3560`
- breakout_eval: `{'threshold': 2.536718785762787, 'accuracy': 0.516809127216053, 'precision': 0.5011026178718798, 'recall': 0.4657236263045392, 'f1': 0.4827658118544103, 'event_rate': 0.4657236263045392, 'hard_negative_rate': 0.6713189113747383, 'signal_frequency': 0.4499987823884668, 'label_frequency': 0.4841832261835184}`
- reversion_eval: `{'threshold': 1.275428056716919, 'accuracy': 0.61152712838496, 'precision': 0.43835486707704796, 'recall': 0.5923833552727804, 'f1': 0.5038604784971737, 'event_rate': 0.5923833552727804, 'hard_negative_rate': 0.605334985191351, 'signal_frequency': 0.4499987823884668, 'label_frequency': 0.33299240210403275}`

## By Timeframe

- `15m` | breakout_f1=`0.4861` | reversion_f1=`0.5013` | corr=`0.3027/0.3453`
- `60m` | breakout_f1=`0.4918` | reversion_f1=`0.5117` | corr=`0.2333/0.3744`
- `1d` | breakout_f1=`0.5064` | reversion_f1=`0.5124` | corr=`0.1650/0.4197`

## By Asset

- `600036` | breakout_f1=`0.4677` | reversion_f1=`0.4989` | corr=`0.2777/0.3692`
- `601166` | breakout_f1=`0.4855` | reversion_f1=`0.4890` | corr=`0.2696/0.3513`
- `600030` | breakout_f1=`0.4845` | reversion_f1=`0.5213` | corr=`0.2788/0.3453`
- `601318` | breakout_f1=`0.4578` | reversion_f1=`0.4895` | corr=`0.2950/0.3388`
- `600519` | breakout_f1=`0.4789` | reversion_f1=`0.4672` | corr=`0.2587/0.3203`
- `000858` | breakout_f1=`0.4847` | reversion_f1=`0.4686` | corr=`0.3205/0.3175`
- `600887` | breakout_f1=`0.4741` | reversion_f1=`0.5295` | corr=`0.2837/0.3953`
- `000333` | breakout_f1=`0.4761` | reversion_f1=`0.5398` | corr=`0.3168/0.4013`
- `600690` | breakout_f1=`0.4792` | reversion_f1=`0.5093` | corr=`0.3118/0.3898`
- `600309` | breakout_f1=`0.4797` | reversion_f1=`0.4975` | corr=`0.3455/0.3401`
- `601899` | breakout_f1=`0.5061` | reversion_f1=`0.4995` | corr=`0.4067/0.3351`
- `600031` | breakout_f1=`0.4872` | reversion_f1=`0.5172` | corr=`0.3453/0.3473`
- `600900` | breakout_f1=`0.4937` | reversion_f1=`0.5156` | corr=`0.2735/0.3868`
- `600028` | breakout_f1=`0.5042` | reversion_f1=`0.4953` | corr=`0.1639/0.3691`
- `300750` | breakout_f1=`0.4796` | reversion_f1=`0.5018` | corr=`0.3098/0.3489`
- `002594` | breakout_f1=`0.4904` | reversion_f1=`0.4940` | corr=`0.3519/0.3148`
- `002475` | breakout_f1=`0.4707` | reversion_f1=`0.4973` | corr=`0.3494/0.3314`
- `002415` | breakout_f1=`0.4830` | reversion_f1=`0.4965` | corr=`0.2683/0.3530`
- `300059` | breakout_f1=`0.4762` | reversion_f1=`0.5123` | corr=`0.2673/0.3589`
- `600276` | breakout_f1=`0.4987` | reversion_f1=`0.5214` | corr=`0.3433/0.3637`
- `300760` | breakout_f1=`0.4965` | reversion_f1=`0.5056` | corr=`0.3178/0.3607`
- `300124` | breakout_f1=`0.4883` | reversion_f1=`0.5338` | corr=`0.3043/0.3938`
- `601012` | breakout_f1=`0.4912` | reversion_f1=`0.5271` | corr=`0.3096/0.3782`
- `603288` | breakout_f1=`0.4989` | reversion_f1=`0.4855` | corr=`0.2646/0.3602`

## Probe Rules

- breakout_threshold: `3.139857`
```text
|--- Volatility <= 0.00
|   |--- EKF_Res <= 0.00
|   |   |--- EKF_Res <= -0.01
|   |   |   |--- class: 0
|   |   |--- EKF_Res >  -0.01
|   |   |   |--- class: 1
|   |--- EKF_Res >  0.00
|   |   |--- EKF_Res <= 0.00
|   |   |   |--- class: 0
|   |   |--- EKF_Res >  0.00
|   |   |   |--- class: 0
|--- Volatility >  0.00
|   |--- Volatility <= 0.00
|   |   |--- MLE <= 0.08
|   |   |   |--- class: 0
|   |   |--- MLE >  0.08
|   |   |   |--- class: 0
|   |--- Volatility >  0.00
|   |   |--- Volatility <= 0.00
|   |   |   |--- class: 0
|   |   |--- Volatility >  0.00
|   |   |   |--- class: 0

```
- reversion_threshold: `7.669978`
```text
|--- EMA_Div <= 0.00
|   |--- EKF_Res <= -0.01
|   |   |--- Entropy <= 0.54
|   |   |   |--- class: 1
|   |   |--- Entropy >  0.54
|   |   |   |--- class: 0
|   |--- EKF_Res >  -0.01
|   |   |--- EMA_Div <= 0.00
|   |   |   |--- class: 0
|   |   |--- EMA_Div >  0.00
|   |   |   |--- class: 0
|--- EMA_Div >  0.00
|   |--- Entropy <= 0.60
|   |   |--- Price_Mom <= -0.00
|   |   |   |--- class: 1
|   |   |--- Price_Mom >  -0.00
|   |   |   |--- class: 1
|   |--- Entropy >  0.60
|   |   |--- Price_Mom <= -0.00
|   |   |   |--- class: 0
|   |   |--- Price_Mom >  -0.00
|   |   |   |--- class: 1

```
