from DataMine import DataMine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from adtk.detector import LevelShiftAD, InterQuartileRangeAD, PersistAD
from adtk.data import validate_series
from adtk.visualization import plot
import warnings
from tqdm import tqdm


def unixToTime(unixtime):
    return pd.to_datetime(unixtime,unit='s',utc=True)


class DataMineTools(DataMine): #继承DataMine
    def ARIMA(self, timestamps, datas): #异常点检测
        dic = {'timestamp' : timestamps,
                'datas' : datas}
        data = pd.DataFrame(dic['datas'],dic['timestamp'])
        data.columns=['datas']

        #画出时序图
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False #用来正常显示表示负号
        data.plot()
        plt.savefig('image/1.时序数据图.jpg')
        
        #画出自相关性图
        plot_acf(data)
        plt.savefig('image/2.自相关性图.jpg')
        
        #平稳性检测
        print(u'【计算】原始序列的检验结果为：\n',adfuller(data[u'datas']))
        #返回值依次为：adf, pvalue p值, usedlag, nobs, critical values临界值 , icbest, regresults, resstore
        #adf 分别大于3中不同检验水平的3个临界值,单位检测统计量对应的p 值显著大于 0.05 , 说明序列可以判定为 非平稳序列
        
        #对数据进行差分后得到 自相关图和 偏相关图
        D_data = data.diff().dropna()
        D_data.columns = [u'差分']
        
        D_data.plot() #画出差分后的时序图
        plt.savefig('image/3.差分时序数据图.jpg')
        
        plot_acf(D_data) #画出自相关图
        plt.savefig('image/4.差分自相关性图.jpg')
        plot_pacf(D_data) #画出偏相关图
        plt.savefig('image/5.差分偏相关性图.jpg')
        print(u'【计算】差分序列的ADF检验结果为：\n', adfuller(D_data[u'差分'])) #平稳性检验
        #差分序列的ADF 检验结果为： (-3.1560562366723537, 0.022673435440048798, 0, 35, {'1%': -3.6327426647230316,
        # '10%': -2.6130173469387756, '5%': -2.9485102040816327}, 287.5909090780334)
        #一阶差分后的序列的时序图在均值附近比较平稳的波动, 自相关性有很强的短期相关性, 
        # 单位根检验 p值小于 0.05 ,所以说一阶差分后的序列是平稳序列
        
        #对一阶差分后的序列做白噪声检验
        print(u'【计算】差分序列的白噪声检验结果：\n',acorr_ljungbox(D_data, lags= 1)) #返回统计量和 p 值
        # 差分序列的白噪声检验结果： (array([11.30402222]), array([0.00077339])) p值为第二项, 远小于 0.05
        
        #对模型进行定阶
        pmax = 5 # int(len(D_data) / 10) #一般阶数不超过 length /10
        qmax = 5 # int(len(D_data) / 10)
        bic_matrix = []
        content = 0
        all = (pmax+1)*(qmax+1)
        train_data = tuple(data['datas'])
        warnings.filterwarnings('ignore')
        print("【计算】ARIMA定阶中,进度：")
        pbar = tqdm(total=all)
        for p in range(pmax +1):
            temp= []
            for q in range(qmax+1):
                try:
                    model = sm.tsa.arima.ARIMA(train_data, order=(p,1,q))
                    temp.append(model.fit().bic)
                    content = content + 1
                except:
                    temp.append(None)
                pbar.update(1)
            bic_matrix.append(temp)
        pbar.close()
        print(temp)
        bic_matrix = pd.DataFrame(bic_matrix) #将其转换成Dataframe 数据结构
        p,q = bic_matrix.stack().idxmin() #先使用stack 展平, 然后使用 idxmin 找出最小值的位置
        print(u'【计算】BIC最小的p值和q值：%s,%s' %(p,q)) # BIC 最小的p值 和 q 值：0,1
    
        #所以可以建立ARIMA 模型,ARIMA(0,1,1)
        model = sm.tsa.arima.ARIMA(train_data, order=(p,1,q)).fit()
        print("【结果】模型报告：", model.summary()) #生成一份模型报告
        print("【结果】预测10个数据：", model.forecast(10)) #为未来5天进行预测,返回预测结果、标准误差和置信区间

        return model.forecast(10)
    
    def ADTK(self, timestamps, datas):#ADTK异常检测
        timestamps = list(map(unixToTime, timestamps))
        dic = {'timestamp' : timestamps,
                'datas' : datas}
        data = pd.DataFrame(dic['datas'],dic['timestamp'])
        data.columns=['datas']
        data = validate_series(data)
        
        level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
        anomalies = level_shift_ad.fit_detect(data)
        im = plot(data, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
        im[0].plot()
        plt.savefig('detect/1.突变性异常图.jpg')
        print("【计算】已生成图片——突变性异常图")
    
        iqr_ad = InterQuartileRangeAD(c=1.5)
        anomalies = iqr_ad.fit_detect(data)
        im = plot(data, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
        im[0].plot()
        plt.savefig('detect/2.上下限异常图.jpg')
        print("【计算】已生成图片——上下限异常图")
        
        persist_ad = PersistAD(c=3.0, side='positive')
        anomalies = persist_ad.fit_detect(data)
        im = plot(data, anomaly=anomalies, anomaly_color='red');
        im[0].plot()
        plt.savefig('detect/3.区间异常图.jpg')
        print("【计算】已生成图片——区间异常图")
        return data


if __name__ == '__main__':
    server = DataMineTools()
    server.run() #开启服务
