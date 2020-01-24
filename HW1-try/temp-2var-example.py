import pandas as pd
import numpy as np

# 从csv中读取有用的信息
df = pd.read_csv('train.csv', usecols=range(3, 27), encoding = 'gb18030')    #取后面24列数据，对应24小时
x_list, y_list = [], []  # df替换指定元素，将空数据填充为0
df = df.replace(['NR'], [0.0])  # astype() 转换array中元素数据类型
array = np.array(df).astype(float)  # 将数据集拆分为多个数据帧

for i in range(0, 4320, 18*20):             #一共12个月，每个月一循环
    for j in range(480-9):                   #24*20=480   480-10+1=471.每个月有471条数据帧及其对应的标签
        if j % 24 <= 14 :                    #判断标签需不需要换天，当标签也在当天
            mat = array[i:i+18,j%24:j%24+9]
            label = array[i+9,j%24+9]
            x_list.append(mat)
            y_list.append(label)
        elif j % 24 == 15:                   #当标签为第二天0点
            mat = array[i:i+18,j%24:24]
            label = array[i + 27, 0]
            x_list.append(mat)
            y_list.append(label)
        else:                                 #当标签为第二天1点后
            mat = np.hstack((array[i:i + 18, j%24:24], array[i + 18:i + 36, 0:j%24 - 15]))
            label = array[i+27,j%24-15]
            x_list.append(mat)
            y_list.append(label)

x = np.array(x_list)                      #数据帧18*9
y = np.array(y_list)                        #标签
print(x.shape)


# 随机划分训练集与验证集，前8/9数据帧作为训练集后1/9个数据帧作为验证集
rand_x = np.arange(x.shape[0])            #取第一列,即每个数据帧的编号，用于后面随机打乱
np.random.shuffle(rand_x)                 #打乱5652个数据帧

x_train, y_train = x[rand_x[0:5024]], y[rand_x[0:5024]]
x_val, y_val = x[rand_x[5024:5652]], y[rand_x[5024:5652]]
epoch = 2000  # 训练轮数


# 开始训练
bias = 0  # 偏置值初始化
weights = np.ones(9)  # 权重初始化，9个pm2.5值的权重
learning_rate = 1  # 初始学习率
reg_rate = 0.001  # 正则项系数
bg2_sum = 0  # 用于存放偏置值的梯度平方和
wg2_sum = np.zeros(9)  # 用于存放权重的梯度平方和

for i in range(epoch):
    b_g = 0
    w_g = np.zeros(9)
    # 在所有数据上计算Loss_label的梯度
    for j in range(5024):
        b_g += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-1)          #x_train[j,9,:]表示第j个数据帧的pm2.5行的全部9个值
        for k in range(9):                                                         #9个w权重值不同（Adagrad方法)
            w_g[k] += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) * (-x_train[j, 9, k])
            # 求平均
    b_g /= 5024    #Loss对b的偏微分
    w_g /= 5024    #Loss对w的偏微分
    #  加上Loss_regularization在w上的梯度
    for m in range(9):
        w_g[m] += reg_rate * weights[m]

    # adagrad
    bg2_sum += b_g ** 2
    wg2_sum += w_g ** 2
    # 更新权重和偏置
    bias -= learning_rate / bg2_sum ** 0.5 * b_g
    weights -= learning_rate / wg2_sum ** 0.5 * w_g

    # 每训练200轮，输出一次在训练集上的损失
    if i % 200 == 0:
        loss = 0
        for j in range(5024):
            loss += (y_train[j] - weights.dot(x_train[j, 9, :]) - bias) ** 2
        print('after {} epochs, the loss on train data is:'.format(i), loss / 5024)


"""
# 在验证集上看效果
loss = 0
for i in range(628):
    loss += (y_val[i] - weights.dot(x_val[i, 9, :]) - bias) ** 2
loss = validate(x_val, y_val, w, b)
print('The loss on val data is:', loss)
"""