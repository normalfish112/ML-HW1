##Import package
import numpy as np
import csv

##Read in training set
raw_data = np.genfromtxt('train.csv', delimiter=',') ## train.csv
data = raw_data[1:,3:]
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0 
month_to_data = {}  ## Dictionary (key:month , value:data)                                  
 
for month in range(12):
    sample = np.empty(shape = (18 , 480))
    for day in range(20):
        for hour in range(24): 
            sample[:,day * 24 + hour] = data[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
    month_to_data[month] = sample
    
##Preprocess
x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
y = np.empty(shape = (12 * 471 , 1),dtype = float)

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue  
            x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1) 
            y[month * 471 + day * 24 + hour,0] = month_to_data[month][9 ,day * 24 + hour + 9]
            
##Normalization
mean = np.mean(x, axis = 0)
std = np.std(x, axis = 0)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0 :
            x[i][j] = (x[i][j]- mean[j]) / std[j]

##Training
dim = x.shape[1] + 1 
w = np.zeros(shape = (dim, 1 ))
x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1).astype(float)
learning_rate_w = np.array([[200]] * dim)
adagrad_sum_w = np.zeros(shape = (dim, 1 ))

bias = 0
learning_rate_b = np.array([[50]] * 5652)
adagrad_sum_b = np.zeros(shape = (5652, 1 ))


for T in range(10000):
    if(T % 1000 == 0 ):
        print("T=",T)
        print("Loss:",np.power(np.sum(np.power(bias+x.dot(w) - y, 2 ))/ x.shape[0],0.5))
      
    gradient_b = (-2) * (y-x.dot(w)-bias) #(y-xw)**2 partial by b
    adagrad_sum_b += gradient_b ** 2
    bias = bias - learning_rate_b * gradient_b / (np.sqrt(adagrad_sum_b) + 0.0005)
   
    gradient_w = (-2) * np.transpose(x).dot(y-x.dot(w)-bias) #(y-xw)**2 partial by w
    adagrad_sum_w += gradient_w ** 2
    w = w - learning_rate_w * gradient_w / (np.sqrt(adagrad_sum_w) + 0.0005)

np.save('weight.npy',w)     ## save weight

##Read in testing set
w = np.load('weight.npy')                                   ## load weight
test_raw_data = np.genfromtxt('test.csv', delimiter=',')   ## test.csv
test_data = test_raw_data[:, 2: ]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0 

##Predict
test_x = np.empty(shape = (240, 18 * 9),dtype = float)

for i in range(240):
    test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 

for i in range(test_x.shape[0]):        ##Normalization
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
answer = test_x.dot(w)+bias #dimension error

##Write file
f = open('prediction_0.0001.csv',"w",newline='')
w = csv.writer(f)
title = ['id','value']
w.writerow(title)  

for i in range(240):
    content = ['id_'+str(i),answer[i][0]]
    w.writerow(content) 
f.close()
#"""