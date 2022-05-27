from xml.etree.ElementPath import prepare_descendant
from DataMine import DataMine, tcn
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
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def unixToTime(unixtime):
    return pd.to_datetime(unixtime,unit='s',utc=True)


# 数据集归一化
def get_normal_data(purchase_redeem_seq):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(purchase_redeem_seq[['datas', 'data2']])
    return scaled_data, scaler


# 构造训练集
def get_train_data(origion_scaled_data, feature_scaled_data, divide_train_valid_index, seq_len):
    train_x, train_y = [], []
    normalized_train_feature = feature_scaled_data[0: divide_train_valid_index]
    normalized_train_label = origion_scaled_data[1: divide_train_valid_index + 1]
    for i in range(len(normalized_train_label) - seq_len + 1):
        train_x.append(normalized_train_feature[i: i + seq_len])
        train_y.append(normalized_train_label[i: i + seq_len])

    return train_x, train_y


# 构造验证集
def get_valid_data(origion_scaled_data, feature_scaled_data, divide_train_valid_index, divide_valid_test_index,
                   seq_len):
    valid_x, valid_y = [], []
    normalized_valid_feature = feature_scaled_data[divide_train_valid_index: divide_valid_test_index]
    normalized_valid_label = origion_scaled_data[divide_train_valid_index + 1: divide_valid_test_index + 1]
    for i in range(len(normalized_valid_label) - seq_len + 1):
        valid_x.append(normalized_valid_feature[i: i + seq_len])
        valid_y.append(normalized_valid_label[i: i + seq_len])

    return valid_x, valid_y


# 构造测试集
def get_test_data(origion_scaled_data, feature_scaled_data, divide_valid_test_index, seq_len):
    test_x, test_y = [], []
    normalized_test_feature = feature_scaled_data[divide_valid_test_index - seq_len + 1: -1]
    normalized_test_label = origion_scaled_data[divide_valid_test_index + 1:]
    for i in range(len(normalized_test_label)):
        test_x.append(normalized_test_feature[i: i + seq_len])
    test_y = normalized_test_label

    return test_x, test_y


class DataMineTools(DataMine): #继承DataMine
    def ARIMA(self, timestamps, datas): #异常点检测
        dic = {'timestamp' : timestamps,
                'datas' : datas}
        data = pd.DataFrame(dic['datas'],dic['timestamp'])
        data.columns=['datas']

        #画出时序图
        plt.rcParams['figure.figsize'] = (16.0, 8.0)
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
        D_data.columns = [u'diff']
        
        D_data.plot() #画出差分后的时序图
        plt.savefig('image/3.差分时序数据图.jpg')
        
        plot_acf(D_data) #画出自相关图
        plt.savefig('image/4.差分自相关性图.jpg')
        plot_pacf(D_data) #画出偏相关图
        plt.savefig('image/5.差分偏相关性图.jpg')
        print(u'【计算】差分序列的ADF检验结果为：\n', adfuller(D_data[u'diff'])) #平稳性检验
        #一阶差分后的序列的时序图在均值附近比较平稳的波动, 自相关性有很强的短期相关性, 
        # 单位根检验 p值小于 0.05 ,所以说一阶差分后的序列是平稳序列
        
        #对一阶差分后的序列做白噪声检验
        print(u'【计算】差分序列的白噪声检验结果：\n',acorr_ljungbox(D_data, lags= 1)) #返回统计量和 p 值
        
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
        bic_matrix = pd.DataFrame(bic_matrix) #将其转换成Dataframe 数据结构
        p,q = bic_matrix.stack().idxmin() #先使用stack 展平, 然后使用 idxmin 找出最小值的位置
        print(u'【计算】BIC最小的p值和q值：%s,%s' %(p,q)) # BIC 最小的p值 和 q 值
    
        #所以可以建立ARIMA 模型
        model = sm.tsa.arima.ARIMA(train_data, order=(p,1,q)).fit()
        print("【结果】模型报告：", model.summary()) #生成一份模型报告
        print("【结果】预测10个数据：", model.forecast(10)) #为未来5天进行预测,返回预测结果、标准误差和置信区间

        return str("predict(10)"), model.forecast(10)
    
    def ADTK(self, timestamps, datas):#ADTK异常检测
        timestamps = list(map(unixToTime, timestamps))
        dic = {'timestamp' : timestamps,
                'datas' : datas}
        data = pd.DataFrame(dic['datas'],dic['timestamp'])
        data.columns=['datas']
        data = validate_series(data)
        
        outliers = []
        
        level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
        anomalies = level_shift_ad.fit_detect(data)
        df2 = anomalies.dropna()
        outliers.append(df2[df2['datas']].count())
        im = plot(data, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
        im[0].plot()
        plt.savefig('detect/1.突变性异常图.jpg')
        print("【计算】已生成图片——突变性异常图")
    
        iqr_ad = InterQuartileRangeAD(c=1.5)
        anomalies = iqr_ad.fit_detect(data)
        df2 = anomalies.dropna()
        outliers.append(df2[df2['datas']].count())
        im = plot(data, anomaly=anomalies, ts_markersize=1, anomaly_color='red', anomaly_tag="marker", anomaly_markersize=2)
        im[0].plot()
        plt.savefig('detect/2.上下限异常图.jpg')
        print("【计算】已生成图片——上下限异常图")
        
        persist_ad = PersistAD(c=3.0, side='positive')
        anomalies = persist_ad.fit_detect(data)
        df2 = anomalies.dropna()
        outliers.append(df2[df2['datas']].count())
        im = plot(data, anomaly=anomalies, anomaly_color='red');
        im[0].plot()
        plt.savefig('detect/3.区间异常图.jpg')
        print("【计算】已生成图片——区间异常图")
        return str("outliers by [LevelShitAD, InterQuartileRangeAD, PersistAD]"), outliers
    
    
    def AutoEncoder(self, timestamps, datas):
        # 训练 
        warnings.filterwarnings('ignore')
        print("【训练】模型训练开始：")
        x_data = np.array([i%31536000 for i in timestamps])
        y_data = np.array(datas)
        x, x_test, y, y_test = train_test_split(x_data, y_data, test_size=0.2)
        print("\t训练集大小：{}，测试集大小：{}".format(len(x), len(x_test)))
        input_layer = keras.Input(shape=(1,))
            
        encoded = Dense(8, activation='relu', activity_regularizer='l2')(input_layer) # l2正则化约束
        encoded = Dense(16, activation='relu', activity_regularizer='l2')(encoded)
        encoded = Dense(32, activation='relu', activity_regularizer='l2')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        epochs = 1000
        batch_size = 32
        
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=100,
            mode='min',
            verbose=2
        )]
        #训练
        autoencoder.fit(x,y,validation_data=(x_test,y_test),callbacks=callbacks,epochs=epochs,batch_size=batch_size)
        autoencoder.summary()
        # 加载模型
        print("【信息】正在加载模型...")
        
        pred = autoencoder.predict(x_data)
        pred_y = []
        errors = []

        for i in pred:
            pred_y.append(i[-1])
        
        plt.figure(figsize=(18, 12))
        plt.plot(pred_y, label='ae_data', color='red')
        plt.plot(y_data, label='data', color='blue')
        plt.savefig('ae/1.预测对比图.jpg')
        print("【信息】已生成图片——预测对比图")
        
        for (yy, recon) in zip(y_data, pred_y):
            # 计算预测和真实图片之间的均方差1
            mse = np.mean((yy - recon) ** 2)
            errors.append(mse)
        
        # compute the q-th quantile of the errors which serves as our
        # threshold to identify anomalies -- any data point that our model
        # reconstructed with > threshold error will be marked as an outlier
        thresh = np.quantile(errors, 0.999)
        idxs = np.where(np.array(errors) >= thresh)[0]
        print("【信息】预测均方差为: {}".format(thresh))
        print("【信息】发现了{} 个异常点。".format(len(idxs)))
        
        return str('outliers found'), idxs.tolist()
    
    
    def TCN(self, timestamps, datas):
        warnings.filterwarnings('ignore')
        dic = {'timestamp' : timestamps,
                'datas' : datas,
                'data2' : datas}
        data = pd.DataFrame(dic)
        pd.to_datetime(data['timestamp'], unit='s')
        data = data.set_index('timestamp')

        indexall = len(datas)
        divide_train_valid_index = int((indexall - 11) * 0.7)
        divide_valid_test_index = indexall - 11
        seq_len = 7
        input_channels = 9
        
        purchase_redeem_seq = data
        origion_scaled_data, scaler = get_normal_data(purchase_redeem_seq)
        train_x, train_y = get_train_data(origion_scaled_data, origion_scaled_data, divide_train_valid_index, seq_len)
        valid_x, valid_y = get_valid_data(origion_scaled_data, origion_scaled_data, divide_train_valid_index,
                                        divide_valid_test_index, seq_len)
        test_x, test_y = get_test_data(origion_scaled_data, origion_scaled_data, divide_valid_test_index, seq_len)
        # return "test",[0,0,0]
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)

        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
        valid_dataset = valid_dataset.shuffle(buffer_size=1024).batch(5)

        model = tcn.TemporalConvNet(input_channels=input_channels, layers_channels=[32, 16, 8, 4, 2], kernel_size=3)
        model.compile(optimizer='adam', loss=keras.losses.mean_squared_error, metrics=['mse'])

        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-3,
            patience=100,
            mode='min',
            verbose=2
        )]
        model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks, epochs=1000, verbose=2)
        model.summary()
        test_x = tf.reshape(test_x, [len(test_x), seq_len, 2])
        test_x_pred = model.predict(test_x)
        pred_y = []

        for i in test_x_pred:
            pred_y.append(i[-1])

        inverse_pred_y = scaler.inverse_transform(pred_y)
        inverse_test_y = scaler.inverse_transform(test_y)
        total_redeem_amt_pred = inverse_pred_y[:, 0]
        total_purchase_amt_pred = inverse_pred_y[:, 1]
        total_redeem_amt_value = inverse_test_y[:, 0]
        total_purchase_amt_value = inverse_test_y[:, 1]


        df = pd.DataFrame(data={'pred': total_redeem_amt_pred, 
                                'value': total_redeem_amt_value})

        plt.figure(figsize=(18, 12))
        plt.subplot(211)
        plt.title('total')
        plt.plot(df['pred'], label='pred', color='blue')
        plt.plot(df['value'], label='value', color='red')
        plt.legend(loc='best')

        plt.savefig('tcn/1.预测对比图.jpg')
        print("【信息】已生成图片——预测对比图")
        return str('model.predict(10)'), total_redeem_amt_pred


if __name__ == '__main__':
    server = DataMineTools()
    server.run() #开启服务
