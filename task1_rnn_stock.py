import pandas as pd
import numpy as np
data_train = pd.read_csv('task1_data_train.csv')
data_train.head()

#%%

#获取收盘价格
price_close = data_train.loc[:,'close']
price_close.head()

#%%

#数据可视化
from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.plot(price_close)
plt.title('gzmt price close')
plt.xlabel('time series')
plt.ylabel('price')
plt.show()

#%%

#数据预处理》归一化
price_n = price_close/max(price_close)
print(price_n)

#%%

#数据序列提取方法
def extract_data(data,time_step=10):
    X = []
    y = []
    for i in range(len(data)-time_step):
        X.append([a for a in data[i:i+time_step]])
        y.append(data[i+time_step])
    X = np.array(X)
    X = X.reshape(X.shape[0],X.shape[1],1)
    return X,y

#%%

#方法测试
test_data = [i for i in range(1,10)]
test_step = 5
X,y = extract_data(test_data,test_step)
print(test_data)
print(y)

#%%

#股票价格数据处理
time_step = 10
X,y = extract_data(price_n,time_step)
print(X[0:2,:,:])
print(y)

#%%

#确认数据维度
print(X.shape,len(y))

#%%

#建立模型
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
model = Sequential()
#添加RNN层
model.add(SimpleRNN(units=5,input_shape=(10,1),activation='relu'))
#输出层
model.add(Dense(units=1,activation='linear'))
model.summary()

#%%

#模型配置
model.compile(optimizer='adam',loss='mean_squared_error')

#%%

#模型训练
model.fit(X,y,batch_size=30,epochs=200)

#%%

#结果预测
y_train_predict = model.predict(X)
y_train_predict = y_train_predict*max(price_close)
print(y_train_predict)

#%%

y = [i*max(price_close) for i in y]
print(y)

#%%

#数据可视化
fig2 = plt.figure()
plt.plot(y,label='real price')
plt.plot(y_train_predict,label='predict price')
plt.title('gzmt price close')
plt.xlabel('time series')
plt.ylabel('price')
plt.legend()
plt.show()

#%%

from sklearn.metrics import r2_score
r2_train = r2_score(y,y_train_predict)
print(r2_train)

#%%

#有的小伙伴训练一次以后发现预测出来的结果不理想，很可能是模型进行初始化的时候选取的随机系数不合适，导致梯度下降搜索时遇到了局部极小值
#解决办法：尝试再次建立模型并训练
#多层感知机结构在进行模型求解时，会给定一组随机的初始化权重系数，这种情况是正常的。通常我们可以观察损失函数是否在变小来发现模型求解是否正常

#%%

#测试集数据
data_test = pd.read_csv('task1_data_test.csv')
data_test.head()
price_test = data_test.loc[:,'close']
price_test.head()
#归一化
price_test_n = price_test/max(price_close)
print(price_test_n)

#%%

#测试数据的序列提取
X_test, y_test = extract_data(price_test_n,time_step)
print(X_test.shape,len(y_test))

#%%

#测试数据的1预测
y_test_predict = model.predict(X_test)
y_test_predict = y_test_predict*max(price_close)
print(y_test_predict)

#%%

y_test = [i*max(price_close) for i in y_test]
print(y_test)

#%%

#r2 
r2_test = r2_score(y_test,y_test_predict)
print(r2_test)

#%%

#数据可视化
fig3 = plt.figure()
plt.plot(y_test,label='real price')
plt.plot(y_test_predict,label='predict price')
plt.title('gzmt price close')
plt.xlabel('time series')
plt.ylabel('price')
plt.legend()
plt.show()

#%%

#数据存储
y_test_r = np.array(y_test).reshape(-1,1)
print(y_test_r.shape)
print(y_test_predict.shape)
final_result = np.concatenate((y_test_r,y_test_predict),axis=1)
print(final_result)
final_result_df = pd.DataFrame(final_result,columns=['real price','predict price'])
final_result_df.to_csv('predict_test.csv')