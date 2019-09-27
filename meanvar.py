#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as  plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, confusion_matrix


# In[192]:


#load data
djia_ass_df = pd.read_excel (r'DowJones.xlsx',header=None,sheet_name='Assets_Returns') 
#djia_ind_df = pd.read_excel (r'DowJones.xlsx',header=None,sheet_name='Index_Returns') 
ff49_ass_df = pd.read_excel (r'FF49industries.xlsx',header=None,sheet_name='Assets_Returns') 
#ff49_ind_df = pd.read_excel (r'FF49industries.xlsx',header=None,sheet_name='Index_Returns') 
ftse_ass_df = pd.read_excel (r'FTSE100.xlsx',header=None,sheet_name='Assets_Returns') 
ftse_ind_df = pd.read_excel (r'FTSE100.xlsx',header=None,sheet_name='Index_Returns') 
sp_ass_df = pd.read_excel (r'SP500.xlsx',header=None,sheet_name='Assets_Returns') 
#sp_ind_df = pd.read_excel (r'SP500.xlsx',header=None,sheet_name='Index_Returns') 
nd100_ass_df = pd.read_excel (r'NASDAQ100.xlsx',header=None,sheet_name='Assets_Returns') 
#nd100_ind_df = pd.read_excel (r'NASDAQ100.xlsx',header=None,sheet_name='Index_Returns') 
ndcom_ass_df = pd.read_excel (r'NASDAQComp.xlsx',header=None,sheet_name='Assets_Returns') 
#ndcom_ind_df = pd.read_excel (r'NASDAQComp.xlsx',header=None,sheet_name='Index_Returns') 


# In[202]:


def plot_returns(data, label1, label2):
    mean_lst = list()
    std_lst = list()
    cumrets_lst = list()
    for i in range(data.shape[1]):
        mean_lst.append(np.mean(data[i]))
        std_lst.append(np.std(data[i]))
        cumrets_lst.append(np.cumsum(data[i]))
        
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    # Plot returns
    plt.subplot(1, 2, 1)
    plt.scatter(mean_lst, std_lst, c='k', label='Data')
    plt.xlabel('Actual Mean Returns')
    plt.ylabel('Actual Volatility')
    plt.title(label1)
    #plt.show()

    plt.subplot(1, 2, 2)
    num_bins = 10
    n, bins, patches = plt.hist(mean_lst, num_bins, facecolor='blue')
    plt.title(label2)
    #plt.show()

    return cumrets_lst #mean_lst, std_lst

a1 = plot_returns(ftse_ass_df,'FTSE 100 Stocks','FTSE 100 Stocks Returns Distibution')
a2 = plot_returns(djia_ass_df,'DJIA Stocks','DJIA Stocks Returns Distibution')
plot_returns(ff49_ass_df,'FF49 Index','FF49 Stocks Returns Distibution')
plot_returns(sp_ass_df,'S&P 500 Index','S&P 500 Stocks Returns Distibution')
plot_returns(nd100_ass_df,'NASDAQ 100 Index','NASDAQ 100 Stocks Returns Distibution')
plot_returns(ndcom_ass_df,'NASDAQ Composite','NASDAQ Composite Stocks Returns Distibution')


# In[206]:


#ftse_ass_df.shape


# In[230]:


def cumulative_returns(data):

    for stk in range(data.shape[1]):
        cumrets_lst = list()
        cumrets_lst.append(data[stk][0])
        y = list()
        y.append(0)

        for i in range(1,data.shape[0]):
            y.append(i)
            cumrets_lst.append(data[stk][i]+cumrets_lst[i-1])

        #title
        s = 'Stock ' + str(stk)
   
        plt.plot(y, cumrets_lst, c='k', label='Data')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title(s)
        plt.show()
        
    return

cumulative_returns(ff49_ass_df)


# In[4]:


## We calculate the covariance matrix
#ftse_cov = ftse_ass_df.cov()
#ftse_cov


# In[5]:


#calculated equal weigths
ftse_eq_wght = np.asmatrix([1 / ftse_ass_df.shape[1]] * ftse_ass_df.shape[1])
#ftse_eq_wght.shape


# In[6]:


def portfolio(returns, weights):
    p = np.asmatrix(np.mean(returns, axis=0))  
    w = weights #np.asmatrix(rand_weights(ftse_ass_df.shape[0]))  
    C = np.asmatrix(returns.cov())
    mu = w * p.T  
    sigma = np.sqrt(w * C * w.T)

    return mu, sigma


# In[7]:


ftse_eq_wght_mu, ftse_eq_wght_sigma = portfolio(ftse_ass_df, ftse_eq_wght)
print(ftse_eq_wght_mu)
print(ftse_eq_wght_sigma)


# In[8]:


ftse_eq_wght_sharpe = ftse_eq_wght_mu / ftse_eq_wght_sigma
ftse_eq_wght_sharpe


# In[9]:


######## FTSE Index

#Sharpe ratio
ftse_ind_sharpe = round(np.mean(ftse_ind_df) / np.std(ftse_ind_df),2)
print(ftse_ind_sharpe)

#Avg return
ftse_ind_avgret = round(np.mean(ftse_ind_df),4)
print(ftse_ind_avgret)

#Sortino ratio
ftse_ind_sortino = round(np.mean(ftse_ind_df) / np.std(np.where(ftse_ind_df > 0, 0, ftse_ind_df)),2)
print(ftse_ind_sortino)


# In[10]:


####Paper checks  --  Bruni et al 2016 


# In[11]:


####FTSE Mean-Var Out of sample

ftse_oos_mv = pd.read_excel (r'OutofSamplePortReturns_MeanVar_FTSE100_List.xlsx',header=None) 
#ftse_oos_mv

#Sharpe Ratio
ftse_oos_mv_sharpe = round(np.mean(ftse_oos_mv) / np.std(ftse_oos_mv),2)
print('Sharpe ratio mv ',float(ftse_oos_mv_sharpe))

#Avg return
ftse_oos_mv_avgret = round(np.mean(ftse_oos_mv),4)
print('Avg return mv ',float(ftse_oos_mv_avgret))

#Sortino ratio
ftse_oos_mv_sortino = round(np.mean(ftse_oos_mv) / np.std(np.where(ftse_oos_mv > 0, 0, ftse_oos_mv)),2)
print('Sortino mv ',float(ftse_oos_mv_sortino))


# In[12]:


####FTSE Index Out of sample

ftse_oos = pd.read_excel (r'OutofSampleReturns_Index_FTSE100.xlsx',header=None) 
#print(ftse_oos)

#Sharpe Ratio
ftse_oos_sharpe = round(np.mean(ftse_oos) / np.std(ftse_oos),2)
print('Sharpe ratio ind ',float(ftse_oos_sharpe))

#Avg return
ftse_oos_avgret = round(np.mean(ftse_oos),4)
print('Avg return ind ',float(ftse_oos_avgret))

#Sortino ratio
ftse_oos_sortino = round(np.mean(ftse_oos) / np.std(np.where(ftse_oos > 0, 0, ftse_oos)),2)
print('Sortio ratio ind ',float(ftse_oos_sortino))

#Jensen alpha
ftse_oos_mv1 = np.concatenate((ftse_oos_mv,ftse_oos),axis=1)
ab = pd.DataFrame(ftse_oos_mv1).cov()[0][1]
ftse_oos_mv_jalpha = round(np.mean(ftse_oos_mv)-(ab / np.var(ftse_oos)) * np.mean(ftse_oos),4)
print('Jensen alpha mv ',float(ftse_oos_mv_jalpha))

#Information ratio
ftse_oos_mv_ir = round(np.mean(ftse_oos_mv - ftse_oos) / np.std(ftse_oos_mv - ftse_oos),2)
print('Information ratio mv ',float(ftse_oos_mv_ir))

#Omega ratio
a = ftse_oos_mv - np.mean(ftse_oos)
ftse_oos_mv_omega = round(np.mean(np.where(a < 0, 0, a)) / -np.mean(np.where(a > 0, 0, a)),4)
print('Omega ratio mv ',ftse_oos_mv_omega)


# In[13]:


#Rachev ratio - Index
a = np.sort(ftse_oos.iloc[:,0].values)
cvr_param = int(round((1-0.95) * len(a),0))
ftse_oos_cvar95_a = (1 / ((1-0.95) * len(a)) * np.sum(a[0:cvr_param]) )

b = np.sort(ftse_oos.iloc[:,0].values*-1)
cvr_param2 = int(round((1-0.95) * len(b),0))
ftse_oos_cvar95_b = (1 / ((1-0.95) * len(b)) * np.sum(b[0:cvr_param2]) )
ftse_oos_rachev = round(ftse_oos_cvar95_b / ftse_oos_cvar95_a,2)
print('Rachev ratio ind ',ftse_oos_rachev)

#Rachev ratio - MV
a = np.sort(ftse_oos_mv.iloc[:,0].values)
cvr_param = int(round((1-0.95) * len(a),0))
ftse_oos_mv_cvar95_a = (1 / ((1-0.95) * len(a)) * np.sum(a[0:cvr_param]) )

b = np.sort(ftse_oos_mv.iloc[:,0].values*-1)
cvr_param2 = int(round((1-0.95) * len(b),0))
ftse_oos_mv_cvar95_b = (1 / ((1-0.95) * len(b)) * np.sum(b[0:cvr_param2]) )
ftse_oos_mv_rachev = round(ftse_oos_mv_cvar95_b / ftse_oos_mv_cvar95_a,2)
print('Rachev ratio mv ',ftse_oos_mv_rachev)


# In[ ]:





# In[14]:


#####SVR


# In[15]:


#Normalization

# Get column names first
names = ftse_ass_df.columns

# Create the Scaler object based on range [-1,1]
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

# Fit your data on the scaler object
scaled_ftse_ass_df = scaler.fit_transform(ftse_ass_df)
scaled_ftse_ass_df = pd.DataFrame(scaled_ftse_ass_df, columns=names)

scaled_ftse_ass_df.head()


# In[16]:


#test invere tansform
test_invscale_df = pd.DataFrame(scaler.inverse_transform(scaled_ftse_ass_df), columns=names)
#test_invscale_df


# In[17]:


scaled_ftse_ass_df = scaled_ftse_ass_df.reset_index()
scaled_ftse_ass_df.rename(columns={'index':'week'}, inplace=True)
scaled_ftse_ass_df.head()


# In[38]:


def predict_returns_SVR(dates, returns, x, C_coef, gamma_coef):
    # convert to 1xn dimension
    dates = np.reshape(dates,(len(dates), 1)) 
    x_test = np.reshape(x,(len(x), 1))
    #print(dates)
    #print(returns)
    
    # Define model
    #svr_lin  = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=C_coef, gamma=gamma_coef)  # C=1e3,gamma=0.1
    
    # Fit regression model
    #svr_lin.fit(dates, returns)
    #svr_poly.fit(dates, returns)
    svr_rbf.fit(dates, returns)
    
    # Plot data
    #plt.scatter(dates, returns, c='k', label='Data')
    #plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    #plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
    #plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    #plt.xlabel('Date')
    #plt.ylabel('Return')
    #plt.title('Support Vector Regression')
    #plt.legend()
    #plt.show()
    
    #y_pred_lin = svr_lin.predict(x_test)
    y_pred_rbf = svr_rbf.predict(x_test)
    
    return y_pred_rbf  #y_pred_lin,    #svr_lin.predict(x)[0], svr_rbf.predict(x)[0] #,,  svr_poly.predict(x)[0] # 


# In[39]:



def predict_returns_RF(dates, returns, x, n_est_coef, rnd_coef):
    # convert to 1xn dimension
    dates = np.reshape(dates,(len(dates), 1)) 
    x_test = np.reshape(x,(len(x), 1))
    #print(dates)
    #print(returns)
    
    rf = RandomForestRegressor(n_estimators=n_est_coef, random_state=rnd_coef)   #n_estimators = 1000, random_state = 42
    
    rf.fit(dates, returns)
    
    # Plot data
    #plt.scatter(dates, returns, c='k', label='Data')
    #plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
    #plt.plot(dates, rf.predict(dates), c='r', label='RF Prediction')    
    #plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
    #plt.xlabel('Date')
    #plt.ylabel('Return')
    #plt.title('Random Forest Regression')
    #plt.legend()
    #plt.show()
    
    y_pred_rf = rf.predict(x_test)
    
    return y_pred_rf


# In[20]:


#print(scaled_ftse_ass_df.head())
#scaled_ftse_ass_df[1][1]
#print(scaled_ftse_ass_df.iloc[0:5,1].values)

#pred_returns_rbf = scaled_ftse_ass_df.copy()
#pred_returns_rbf.iloc[:,1:]=0.0
#pred_returns_rbf.iloc[0:5,1] = scaled_ftse_ass_df.iloc[0:5,1].values
#print(pred_returns_rbf.head(10))


# In[122]:


# Find predicted returns (main)

def predict_main(df, use_expost, insample_start, oosample_start, insample_length, oosample_length, rolling_length, 
                 nstocks, nrebs, use_SVR=True, SVR_C=1000, SVR_gamma=0.1, use_RF=False, RF_nest=0, RF_rand=42):

    #insample_start = 0
    #oosample_start = 52
    #insample_length = 52
    #oosample_length = 12
    #rolling_length = 12
    #nstocks = 83
    #nrebs = 56

    #copy dataframe and initialise values
    pred_returns_rbf = df.copy()
    pred_returns_rbf.iloc[:,1:]=0.0
    #print(pred_returns_rbf.head())

    for nn in range(3): #nstocks):
        print("=========")
        print('Stock ',nn)
        #print("=========")

        #initialize in-sample array
        insample_returns_rbf = df[nn]

        pred_returns_rbf.iloc[0:insample_length,nn+1] = df.iloc[0:insample_length,nn+1].values
        #print(pred_returns_rbf.head(60))

        for nwin in range(nrebs):
            #print("     ~~~~~~~~~~")
            print('     Window ',nwin)
            #print("     ~~~~~~~~~~")

            #advance insample and o-o-sample start positions by rolling length
            if nwin>0:
                insample_start = nwin*rolling_length
                oosample_start = insample_length + (nwin*rolling_length)

            for i in range(oosample_length):
                #print('          Iteration ',i+1) #, insample_start, oosample_start)

                #set in-sample data
                weeks = scaled_ftse_ass_df['week'][i+insample_start:i+oosample_start,].tolist()
                #print(i+insample_start,i+oosample_start,len(weeks),weeks)

                if use_expost==True:
                    #ex-post returns
                    insample_returns_p = df[nn][i+insample_start:i+oosample_start,].tolist() 
                    #print(scaled_ftse_ass_df[nn][i+insample_start:i+oosample_start,].tolist())
                    #print(len(insample_returns_p),insample_returns_p)

                    #predict one-step ahead return using ex-post (actual) returns
                    if use_SVR==True:
                        outofsample_predicted_returns = predict_returns_SVR(weeks, insample_returns_p, [i+oosample_start], SVR_C, SVR_gamma)
                    else:
                        outofsample_predicted_returns = predict_returns_RF(weeks, insample_returns_p, [i+oosample_start], RF_nest, RF_rand)
                    
                    #print('Predicted returns for week:',i+oosample_start)
                    #print(len(outofsample_predicted_returns),outofsample_predicted_returns)
                else:
                    #ex-ante returns
                    insample_returns = insample_returns_rbf[i+insample_start:i+oosample_start,].tolist() 
                    #print(insample_returns_rbf[insample_start:i+oosample_start,].tolist())

                    #predict one-step ahead return using ex-ante (predicted) returns
                    if use_SVR==True:
                        outofsample_predicted_returns = predict_returns_SVR(weeks, insample_returns, [i+oosample_start], SVR_C, SVR_gamma)
                                                        #scaled_ftse_ass_df[nn][i+oosample_start].tolist()) 
                    else:
                        outofsample_predicted_returns = predict_returns_RF(weeks, insample_returns, [i+oosample_start], RF_nest, RF_rand)
                    
                #add result to predicted returns dataframe
                pred_returns_rbf.iloc[i+oosample_start,nn+1] = outofsample_predicted_returns[0]
                #print(pred_returns_rbf.head(60))


                #insample_returns_lin = np.concatenate((np.array(insample_returns),outofsample_predicted_returns[0]),axis=0)
                #add predicted return to in-sample for the next iteration
                insample_returns_rbf = np.concatenate((np.array(insample_returns_rbf[0:i+oosample_start,]),
                                                       outofsample_predicted_returns),axis=0)

                #print('Rolling returns with SVR(rbf):')
                #print(len(insample_returns_rbf),insample_returns_rbf)

    return pred_returns_rbf


# In[123]:


# Set training parameters
insample_start = 0
oosample_start = 100
insample_length = 100
oosample_length = 4
rolling_length = 4

# For FTSE100
nstocks = 83
nrebs = 56


# In[127]:


# Run time approx 40 mins

#C=1000
#gamma=0.1
returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                               insample_length, oosample_length, rolling_length, nstocks, nrebs,
                               use_SVR=True, SVR_C=1000, SVR_gamma=0.1, use_RF=False)
#returns_svm_rbf

#C=1000
#gamma=0.3
#returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                               insample_length, oosample_length, rolling_length, nstocks, nrebs,
#                               use_SVR=True, SVR_C=1000, SVR_gamma=0.3, use_RF=False)
#returns_svr_rbf2

#C=1000
#gamma=0.5
#returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                               insample_length, oosample_length, rolling_length, nstocks, nrebs,
#                               use_SVR=True, SVR_C=1000, SVR_gamma=0.5, use_RF=False)
#returns_svr_rbf3

#C=10000
#gamma=0.2
#returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                               insample_length, oosample_length, rolling_length, nstocks, nrebs,
#                               use_SVR=True, SVR_C=10000, SVR_gamma=0.2, use_RF=False)
#returns_svr_rbf4


# In[ ]:





# In[ ]:





# In[ ]:





# In[94]:


#C=1000
#gamma=0.1
returns_svr_rbf5 = predict_main(scaled_ftse_ass_df, False, 0, 52, 52, 12, 12, 83, 56, 1000, 0.1)
#returns_svr_rbf5

#C=1000
#gamma=0.1
returns_svr_rbf6 = predict_main(scaled_ftse_ass_df, False, 0, 52, 52, 12, 12, 83, 56, 1000, 0.3)
#returns_svr_rbf6

#C=1000
#gamma=0.1
returns_svr_rbf7 = predict_main(scaled_ftse_ass_df, False, 0, 52, 52, 12, 12, 83, 56, 1000, 0.5)
#returns_svr_rbf7

#C=10000
#gamma=0.2
returns_svr_rbf8 = predict_main(scaled_ftse_ass_df, False, 0, 52, 52, 12, 12, 83, 56, 10000, 0.2)
#returns_svr_rbf8

#C=100000
#gamma=0.2
returns_svr_rbf9 = predict_main(scaled_ftse_ass_df, False, 0, 52, 52, 12, 12, 83, 56, 100000, 0.2)
#returns_svr_rbf9


# In[138]:


#n_est=100
returns_rf1 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                           insample_length, oosample_length, rolling_length, nstocks, nrebs,
                           use_SVR=False, use_RF=True, RF_nest=100)
#returns_rf1

#n_est=300
returns_rf2 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                           insample_length, oosample_length, rolling_length, nstocks, nrebs,
                           use_SVR=False, use_RF=True, RF_nest=50)
#returns_svr_rbf2

#n_est=500
returns_rf3 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                           insample_length, oosample_length, rolling_length, nstocks, nrebs,
                           use_SVR=False, use_RF=True, RF_nest=500)
#returns_svr_rbf3

#n_est=1000
returns_rf4 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                           insample_length, oosample_length, rolling_length, nstocks, nrebs,
                           use_SVR=False, use_RF=True, RF_nest=1000)
#returns_svr_rbf4


# In[130]:


def predict_error(model, act_df, pred_df, nstocks, start_index, end_index):
    print('Regression metrics for ',model,':')

    for nn in range(3):  #nstocks):
        print("=========")
        print('Stock ',nn)
        print("=========")

        y_actual = act_df[nn][start_index:end_index] #insample_length+oosample_length]
        y_predicted = pred_df[nn][start_index:end_index] #insample_length+oosample_length]

        rms = np.sqrt(mean_squared_error(y_actual, y_predicted))
        print('RMSE ',rms)
        mae = mean_absolute_error(y_actual, y_predicted)
        print('MAE ',mae)
        mdae = median_absolute_error(y_actual, y_predicted)
        print('MedAE ',mdae)
        
        #print('Confusion matrix:')
        #cm = confusion_matrix(y_actual,y_predicted)
        #print(cm)
    return rms,mae,mdae


# In[102]:


# Plot data
#plt.scatter(dates, returns, c='k', label='Data')
plt.plot(scaled_ftse_ass_df['week'][52:711], scaled_ftse_ass_df[0][52:711], c='g', label='actual')
#plt.plot(scaled_ftse_ass_df['week'][52:711], returns_svm_rbf[0][52:711], c='r', label='predicted')    
#plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Actual returns normalized [-1,1]')
plt.legend()
plt.show()

plt.plot(scaled_ftse_ass_df['week'][52:711], returns_svr_rbf4[0][52:711], c='r', label='predicted')    
#plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Support Vector Regression predicted returns')
plt.legend()
plt.show()


# In[136]:


#measure prediction error
predict_error('SVR(rbf)', scaled_ftse_ass_df, returns_svr_rbf, nstocks, oosample_start, 711)
predict_error('SVR(rbf)2', scaled_ftse_ass_df, returns_svr_rbf2, nstocks, 52, 711)
predict_error('SVR(rbf)3', scaled_ftse_ass_df, returns_svr_rbf3, nstocks, 52, 711)
predict_error('SVR(rbf)4', scaled_ftse_ass_df, returns_svr_rbf4, nstocks, 52, 711)

predict_error('SVR(rbf)5', scaled_ftse_ass_df, returns_svr_rbf5, nstocks, 52, 711)
predict_error('SVR(rbf)6', scaled_ftse_ass_df, returns_svr_rbf6, nstocks, 52, 711)
predict_error('SVR(rbf)7', scaled_ftse_ass_df, returns_svr_rbf7, nstocks, 52, 711)
predict_error('SVR(rbf)8', scaled_ftse_ass_df, returns_svr_rbf8, nstocks, 52, 711)
predict_error('SVR(rbf)9', scaled_ftse_ass_df, returns_svr_rbf9, nstocks, 52, 711)



# In[139]:


predict_error('RF', scaled_ftse_ass_df, returns_rf1, nstocks, oosample_start, 711)
predict_error('RF2', scaled_ftse_ass_df, returns_rf2, nstocks, oosample_start, 711)
predict_error('RF3', scaled_ftse_ass_df, returns_rf3, nstocks, oosample_start, 711)
predict_error('RF4', scaled_ftse_ass_df, returns_rf4, nstocks, oosample_start, 711)


# In[ ]:


#weeks = scaled_ftse_ass_df['week'][0:insample_length,].tolist()
#insample_returns = scaled_ftse_ass_df[0][0:insample_length,].tolist() #X-train = 52 weeks
#print(weeks,returns)

#outofsample_predicted_returns = predict_returns_SVR(weeks, insample_returns, 
#                                                    scaled_ftse_ass_df[0][52:64,].tolist()) # [52:63])
#outofsample_predicted_returns

#insample_returns
#concatenate predicted returns to in sample returns for the next window
#insample_returns_lin = np.concatenate((np.array(insample_returns),outofsample_predicted_returns[0]),axis=0)
#insample_returns_rbf = np.concatenate((np.array(insample_returns),outofsample_predicted_returns[1]),axis=0)


# In[ ]:


#ftse_nreb = 56
#
#insample_returns_rbf = scaled_ftse_ass_df[0]
#
#for i in range(4):   #oosample_length
#    insample_start = i*rolling_length
#    oosample_start = i*rolling_length+insample_length
#    print(i, insample_start, oosample_start)
#
#    #set in-sample data
#    weeks = scaled_ftse_ass_df['week'][insample_start:oosample_start,].tolist()
#    insample_returns = insample_returns_rbf[insample_start:oosample_start,].tolist() #X-train = 52 weeks
#    print('In sample returns:')
#    print(insample_returns)
#   
#    #SVR predictions
#    print(insample_start, oosample_start, oosample_start+oosample_length)
#    print(scaled_ftse_ass_df[0][oosample_start:oosample_start+oosample_length,].tolist())
#    outofsample_predicted_returns = predict_returns_RF(weeks, insample_returns, 
#                                                        scaled_ftse_ass_df[0][oosample_start:oosample_start+oosample_length,].tolist()) 
#    print('Predicted returns:')
#    print(outofsample_predicted_returns)
#
#    #concatenate predicted returns to in sample returns for the next iteration
#    #insample_returns_lin = np.concatenate((np.array(insample_returns),outofsample_predicted_returns[0]),axis=0)
#    insample_returns_rbf = np.concatenate((np.array(insample_returns_rbf[0:oosample_start,]),
#                                           outofsample_predicted_returns),axis=0)
#
#    print('Rolling in sample returns:')
#    print(insample_returns_rbf)


# In[ ]:





# In[ ]:


#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Fit regression model
#svr_lin.fit(weeks, returns)
#svr_poly.fit(weeks, returns)
#svr_rbf.fit(weeks, returns)

#plt.scatter(weeks, returns, c='k', label='Data')
#plt.plot(weeks, svr_lin.predict(weeks), c='g', label='Linear model')
#plt.plot(weeks, svr_rbf.predict(weeks), c='r', label='RBF model')    
#plt.plot(weeks, svr_poly.predict(weeks), c='b', label='Polynomial model')
#plt.xlabel('Weeks')
#plt.ylabel('Returns')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()
    
#print(svr_rbf.predict(x)[0])
#print(svr_lin.predict(x)[0])
#print(svr_poly.predict(x)[0])
    


#_1stcol = scaled_ftse_ass_df[1]
#X = _1stcol.iloc[0:52].values.astype(float)
#y = _1stcol.iloc[53].values.astype(float)
#_1stcol

#regressor = SVR(kernel='rbf')
#regressor.fit(X,y)

#plt.scatter(X, y, color = 'magenta')
#plt.plot(X, regressor.predict(X), color = 'green')
#plt.title('')
#plt.xlabel('Week')
#plt.ylabel('Return')
#plt.show()


# In[ ]:





# In[ ]:




