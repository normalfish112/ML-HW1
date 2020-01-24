##Import package
import sys
import numpy as np
import pandas as pd
import csv

##Read in training set
raw_data = np.genfromtxt('train.csv', delimiter=',') ## train.csv
data_1 = raw_data[1:,3:]

data = np.empty(shape = (20*12*8 , 24),dtype = float)
for i in range(240):
    data[8*i+0,:] = data_1[18*i+2,:] #CO
    data[8*i+1,:] = data_1[18*i+4,:] #NO
    data[8*i+2,:] = data_1[18*i+5,:] #NO2
    data[8*i+3,:] = data_1[18*i+6,:] #NOx
    data[8*i+4,:] = data_1[18*i+8,:] #pm10
    data[8*i+5,:] = data_1[18*i+9,:] #pm2.5
    data[8*i+6,:] = data_1[18*i+10,:] #rainfall
    data[8*i+7,:] = data_1[18*i+12,:] #SO2


where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0 
month_to_data = {}  ## Dictionary (key:month , value:data)                                  
 
for month in range(12):
    sample = np.empty(shape = (8 , 480))
    for day in range(20):
        for hour in range(24): 
            sample[:,day * 24 + hour] = data[8 * (month * 20 + day): 8 * (month * 20 + day + 1),hour]
    month_to_data[month] = sample
    
##Preprocess
x = np.empty(shape = (12 * 471 , 8 * 9),dtype = float)
y = np.empty(shape = (12 * 471 , 1),dtype = float)

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue  
            x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1) 
            y[month * 471 + day * 24 + hour,0] = month_to_data[month][5 ,day * 24 + hour + 9]
            
##Normalization
mean = np.mean(x, axis = 0)
std = np.std(x, axis = 0)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0 :
            x[i][j] = (x[i][j]- mean[j]) / std[j]

##validation
x_val = np.empty((9,628,72))
y_val = np.empty((9,628,1))
x_train = np.empty((9,5024,72))
y_train = np.empty((9,5024,1))
x1_train = np.empty((9,5024,73))
x1_val = np.empty((9,628,73))
#y1_val = np.empty((9,628,1))

w = np.empty((9,73,1))
gradient = np.empty((9,73,1))
#W = np.zeros(shape = (73, 1 ))

adagrad_sum = {}
for i in range(9):
    x_val[i]=x[628*i:628*i+628]                              #每次取一份作为验证集
    y_val[i]=y[628*i:628*i+628]
    x_train[i]=np.vstack((x[0:628*i],x[628*i+628:5652]))     #拼接剩余数据，作为训练集
    y_train[i]=np.vstack((y[0:628*i],y[628*i+628:5652]))

    dim = 73
    w[i] = np.zeros(shape = (dim, 1 ))

    x1_train[i] = np.concatenate((np.ones((x_train[i].shape[0], 1 )), x_train[i]) , axis = 1).astype(float)
    x1_val[i] = np.concatenate((np.ones((x1_val[i].shape[0], 1 )), x_val[i]) , axis = 1).astype(float)
    learning_rate = np.array([[200]] * dim)
    adagrad_sum[i] = np.zeros(shape = (dim, 1 ))

    for T in range(10000):
        #if(T % 1000 == 0 ):
            #print("T=",T)
            #print("Loss:",np.power(np.sum(np.power(x.dot(w) - y, 2 ))/ x.shape[0],0.5))

        gradient[i] = (-2) * np.transpose(x1_train[i]).dot(y_train[i]-x1_train[i].dot(w[i])) #(y-xw)**2 partial by w
        adagrad_sum[i] += gradient[i] ** 2
        w[i] = w[i] - learning_rate * gradient[i] / (np.sqrt(adagrad_sum[i]) + 0.0005)
    #W += w[i]
    Loss = np.power(np.sum(np.power(x1_val[i].dot(w[i]) - y_val[i], 2 ))/ x1_val[i].shape[0],0.5)
    print('i=',i,'Loss=',Loss)
    

W = np.zeros(shape = (73, 1 ))
W = w[4]
#W = W/ 9

np.save('weight.npy',W)     ## save weight

##Read in testing set
w = np.load('weight.npy')                                   ## load weight
test_raw_data = np.genfromtxt('test.csv', delimiter=',')   ## test.csv
test_data_1 = test_raw_data[:, 2: ]

test_data = np.empty(shape = (20*12*8 , 9),dtype = float)
for i in range(240):
    test_data[8*i+0,:] = test_data_1[18*i+2,:] #CO
    test_data[8*i+1,:] = test_data_1[18*i+4,:] #NO
    test_data[8*i+2,:] = test_data_1[18*i+5,:] #No2
    test_data[8*i+3,:] = test_data_1[18*i+6,:] #NOx
    test_data[8*i+4,:] = test_data_1[18*i+8,:] #pm10
    test_data[8*i+5,:] = test_data_1[18*i+9,:] #pm2.5
    test_data[8*i+6,:] = test_data_1[18*i+10,:] #rainfall
    test_data[8*i+7,:] = test_data_1[18*i+12,:] #SO2

where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0 

##Predict
test_x = np.empty(shape = (240, 8 * 9),dtype = float)

for i in range(240):
    test_x[i,:] = test_data[8 * i : 8 * (i+1),:].reshape(1,-1) 

for i in range(test_x.shape[0]):        ##Normalization
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
answer = test_x.dot(W)



##Write file
f = open('prediction.csv',"w",newline='')
w = csv.writer(f)
title = ['id','value']
w.writerow(title)  

for i in range(240):
    content = ['id_'+str(i),answer[i][0]]
    w.writerow(content) 
f.close()
