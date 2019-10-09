#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as  plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, confusion_matrix
import scipy.optimize as sco


# In[2]:


#Load data - asset returns and index returns
djia_ass_df = pd.read_excel (r'DowJones.xlsx',header=None,sheet_name='Assets_Returns') 
djia_ind_df = pd.read_excel (r'DowJones.xlsx',header=None,sheet_name='Index_Returns') 
ff49_ass_df = pd.read_excel (r'FF49industries.xlsx',header=None,sheet_name='Assets_Returns') 
#ff49_ind_df = pd.read_excel (r'FF49industries.xlsx',header=None,sheet_name='Index_Returns') 
ftse_ass_df = pd.read_excel (r'FTSE100.xlsx',header=None,sheet_name='Assets_Returns') 
ftse_ind_df = pd.read_excel (r'FTSE100.xlsx',header=None,sheet_name='Index_Returns') 
sp_ass_df = pd.read_excel (r'SP500.xlsx',header=None,sheet_name='Assets_Returns') 
#sp_ind_df = pd.read_excel (r'SP500.xlsx',header=None,sheet_name='Index_Returns') 
nd100_ass_df = pd.read_excel (r'NASDAQ100.xlsx',header=None,sheet_name='Assets_Returns') 
#nd100_ind_df = pd.read_excel (r'NASDAQ100.xlsx',header=None,sheet_name='Index_Returns') 
#ndcomp_ass_df = pd.read_excel (r'NASDAQComp.xlsx',header=None,sheet_name='Assets_Returns') 
#ndcom_ind_df = pd.read_excel (r'NASDAQComp.xlsx',header=None,sheet_name='Index_Returns') 


# In[3]:


#Plot returns
def plot_returns(data, label1, label2):
    #initialize lists
    mean_lst = list()
    std_lst = list()
    #mean and st dev of returns for each stock
    for i in range(data.shape[1]):
        mean_lst.append(np.mean(data[i]))
        std_lst.append(np.std(data[i]))
        
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

    return #mean_lst, std_lst

plot_returns(ftse_ass_df,'FTSE 100 Stocks','FTSE 100 Stocks Returns Distibution')
plot_returns(djia_ass_df,'DJIA Stocks','DJIA Stocks Returns Distibution')
plot_returns(ff49_ass_df,'FF49 Index','FF49 Stocks Returns Distibution')
plot_returns(sp_ass_df,'S&P 500 Index','S&P 500 Stocks Returns Distibution')
plot_returns(nd100_ass_df,'NASDAQ 100 Index','NASDAQ 100 Stocks Returns Distibution')
#plot_returns(ndcom_ass_df,'NASDAQ Composite','NASDAQ Composite Stocks Returns Distibution')


# In[4]:


#ftse_ass_df.shape


# In[5]:


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

cumulative_returns(djia_ass_df)


# In[ ]:





# In[6]:


#Simple Markowitz test with equal weighted portfolio
#calculated equal weigths
ftse_eq_wght = np.asmatrix([1 / ftse_ass_df.shape[1]] * ftse_ass_df.shape[1])
#ftse_eq_wght.shape

def portfolio(returns, weights):
    p = np.asmatrix(np.mean(returns, axis=0))  
    w = weights #np.asmatrix(rand_weights(ftse_ass_df.shape[0]))  
    C = np.asmatrix(returns.cov())
    mu = w * p.T  
    sigma = np.sqrt(w * C * w.T)

    return mu, sigma

ftse_eq_wght_mu, ftse_eq_wght_sigma = portfolio(ftse_ass_df, ftse_eq_wght)
print(ftse_eq_wght_mu)
print(ftse_eq_wght_sigma)
ftse_eq_wght_sharpe = ftse_eq_wght_mu / ftse_eq_wght_sigma
ftse_eq_wght_sharpe


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


######## FTSE Index

#Sharpe ratio
ftse_ind_sharpe = round(np.mean(ftse_ind_df) / np.std(ftse_ind_df),4)
print(ftse_ind_sharpe)

#Avg return
ftse_ind_avgret = round(np.mean(ftse_ind_df),4)
print(ftse_ind_avgret)

#Sortino ratio
ftse_ind_sortino = round(np.mean(ftse_ind_df) / np.std(np.where(ftse_ind_df > 0, 0, ftse_ind_df)),4)
print(ftse_ind_sortino)


# In[8]:


####Paper checks  --  Bruni et al 2016 


# In[9]:


####FTSE Mean-Var Out of sample

ftse_oos_mv = pd.read_excel (r'OutofSamplePortReturns_MeanVar_FTSE100_List.xlsx',header=None) 
#ftse_oos_mv

#Sharpe Ratio
ftse_oos_mv_sharpe = round(np.mean(ftse_oos_mv) / np.std(ftse_oos_mv),4)
print('Sharpe ratio mv ',float(ftse_oos_mv_sharpe))

#Avg return
ftse_oos_mv_avgret = round(np.mean(ftse_oos_mv),4)
print('Avg return mv ',float(ftse_oos_mv_avgret))

#Sortino ratio
ftse_oos_mv_sortino = round(np.mean(ftse_oos_mv) / np.std(np.where(ftse_oos_mv > 0, 0, ftse_oos_mv)),4)
print('Sortino mv ',float(ftse_oos_mv_sortino))


# In[10]:


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


# In[11]:


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


# In[12]:


######## DJIA Index

#Sharpe ratio
djia_ind_sharpe = round(np.mean(djia_ind_df) / np.std(djia_ind_df),2)
print(djia_ind_sharpe)

#Avg return
djia_ind_avgret = round(np.mean(djia_ind_df),4)
print(djia_ind_avgret)

#Sortino ratio
djia_ind_sortino = round(np.mean(djia_ind_df) / np.std(np.where(djia_ind_df > 0, 0, djia_ind_df)),2)
print(djia_ind_sortino)


# In[13]:


#### DJIA Mean-Var Out of sample

djia_oos_mv = pd.read_excel (r'OutofSamplePortReturns_MeanVar_DowJones_List.xlsx',header=None) 
#djia_oos_mv

#Sharpe Ratio
djia_oos_mv_sharpe = round(np.mean(djia_oos_mv) / np.std(djia_oos_mv),2)
print('Sharpe ratio mv ',float(djia_oos_mv_sharpe))

#Avg return
djia_oos_mv_avgret = round(np.mean(djia_oos_mv),4)
print('Avg return mv ',float(djia_oos_mv_avgret))

#Sortino ratio
djia_oos_mv_sortino = round(np.mean(djia_oos_mv) / np.std(np.where(djia_oos_mv > 0, 0, djia_oos_mv)),2)
print('Sortino mv ',float(djia_oos_mv_sortino))


# In[14]:


#### DJIA Index Out of sample

djia_oos = pd.read_excel (r'OutofSampleReturns_Index_DowJones.xlsx',header=None) 
#print(djia_oos)

#Sharpe Ratio
djia_oos_sharpe = round(np.mean(djia_oos) / np.std(djia_oos),2)
print('Sharpe ratio ind ',float(djia_oos_sharpe))

#Avg return
djia_oos_avgret = round(np.mean(djia_oos),4)
print('Avg return ind ',float(djia_oos_avgret))

#Sortino ratio
djia_oos_sortino = round(np.mean(djia_oos) / np.std(np.where(djia_oos > 0, 0, djia_oos)),2)
print('Sortio ratio ind ',float(djia_oos_sortino))

#Jensen alpha
djia_oos_mv1 = np.concatenate((djia_oos_mv,djia_oos),axis=1)
ab = pd.DataFrame(djia_oos_mv1).cov()[0][1]
djia_oos_mv_jalpha = round(np.mean(djia_oos_mv)-(ab / np.var(djia_oos)) * np.mean(djia_oos),4)
print('Jensen alpha mv ',float(djia_oos_mv_jalpha))

#Information ratio
djia_oos_mv_ir = round(np.mean(djia_oos_mv - djia_oos) / np.std(djia_oos_mv - djia_oos),2)
print('Information ratio mv ',float(djia_oos_mv_ir))

#Omega ratio
a = djia_oos_mv - np.mean(djia_oos)
djia_oos_mv_omega = round(np.mean(np.where(a < 0, 0, a)) / -np.mean(np.where(a > 0, 0, a)),4)
print('Omega ratio mv ',djia_oos_mv_omega)


# In[15]:


#Rachev ratio - Index
a = np.sort(djia_oos.iloc[:,0].values)
cvr_param = int(round((1-0.95) * len(a),0))
djia_oos_cvar95_a = (1 / ((1-0.95) * len(a)) * np.sum(a[0:cvr_param]) )

b = np.sort(djia_oos.iloc[:,0].values*-1)
cvr_param2 = int(round((1-0.95) * len(b),0))
djia_oos_cvar95_b = (1 / ((1-0.95) * len(b)) * np.sum(b[0:cvr_param2]) )
djia_oos_rachev = round(djia_oos_cvar95_b / djia_oos_cvar95_a,2)
print('Rachev ratio ind ',djia_oos_rachev)

#Rachev ratio - MV
a = np.sort(djia_oos_mv.iloc[:,0].values)
cvr_param = int(round((1-0.95) * len(a),0))
djia_oos_mv_cvar95_a = (1 / ((1-0.95) * len(a)) * np.sum(a[0:cvr_param]) )

b = np.sort(djia_oos_mv.iloc[:,0].values*-1)
cvr_param2 = int(round((1-0.95) * len(b),0))
djia_oos_mv_cvar95_b = (1 / ((1-0.95) * len(b)) * np.sum(b[0:cvr_param2]) )
djia_oos_mv_rachev = round(djia_oos_mv_cvar95_b / djia_oos_mv_cvar95_a,2)
print('Rachev ratio mv ',djia_oos_mv_rachev)


# In[16]:


#Load out of sample returns
#### FF49 Industries Index Out of sample
ff49_oos = pd.read_excel (r'OutofSampleReturns_Index_FF49Industries.xlsx',header=None) 
#### S&P500 Index Out of sample
sp_oos = pd.read_excel (r'OutofSampleReturns_Index_SP500.xlsx',header=None) 
#### NASDAQ100 Index Out of sample
nd100_oos = pd.read_excel (r'OutofSampleReturns_Index_NASDAQ100.xlsx',header=None) 
#### NASDAQComposite Out of sample
#ndcomp_oos = pd.read_excel (r'OutofSampleReturns_Index_NASDAQComp.xlsx',header=None) 

#Load Mean-Variance optimial portfolio weights data
ftse_optmv_df = pd.read_excel (r'OptPortfolios_MeanVar_FTSE100.xlsx',header=None)#,sheet_name='Index_Returns') 
djia_optmv_df = pd.read_excel (r'OptPortfolios_MeanVar_DowJones.xlsx',header=None)#,sheet_name='Index_Returns') 
ff49_optmv_df = pd.read_excel (r'OptPortfolios_MeanVar_FF49Industries.xlsx',header=None)#,sheet_name='Index_Returns') 
sp_optmv_df = pd.read_excel (r'OptPortfolios_MeanVar_SP500.xlsx',header=None)#,sheet_name='Index_Returns') 
nd100_optmv_df = pd.read_excel (r'OptPortfolios_MeanVar_NASDAQ100.xlsx',header=None)#,sheet_name='Index_Returns') 
#ndcomp_optmv_df = pd.read_excel (r'OptPortfolios_MeanVar_NASDAQComp.xlsx',header=None)#,sheet_name='Index_Returns')


# In[ ]:





# In[164]:


#Set global parameters for optimization
ftse_nrebs = 56
djia_nrebs = 110
ff49_nrebs = 190
sp_nrebs = 46
nd100_nrebs = 46

ftse_nstocks = 83
djia_nstocks = 28
ff49_nstocks = 49
sp_nstocks = 442
nd100_nstocks = 82

#set rolling window parameters
insample_start = 0
insample_length = 52
oosample_start = 52
oosample_length = 12
rolling_length = 12


# ### Mean-Variance optimization using weights supplied in dataset of Bruni et al 2016

# In[146]:


def _portfolio_meanvar_oosample_returns(weights_df, asset_returns_df, n_rebalances):
    #set rolling window parameters
    insample_start = 0
    rolling_length = 12
    insample_length = 52
    oosample_start = 52
    oosample_length = 12

    #inslen = insample_length
    #rollen = rolling_length
    #ooslen = oosample_length
    #oosstr = oosample_start
    #print(inslen,rollen,ooslen,oosstr)
    #initialize final list
    mvars = list()
    
    #iterate for each rebalance
    for j in range(n_rebalances):
        #for the last rebalance change oosample_lenght depending on how much data is left
        if j==n_rebalances-1:  
            oosample_length=(asset_returns_df.shape[0]-insample_length)%rolling_length
            #ooslen=(asset_returns_df.shape[0]-inslen)%rollen
        
        #move rolling window parameters
        insample_start = j*rolling_length
        oosample_start = insample_length + (j*rolling_length)
        #oosstr = inslen + (j*rollen)

        #iterate for each out of sample week
        for i in range(oosample_length):
            wghts = np.array(weights_df[j])
            rets = np.array(asset_returns_df.iloc[oosample_start+i,]) 
            #rets = np.array(asset_returns_df.iloc[oosstr+i,]) 
            
            #calculate out of sample return
            oosrets = np.dot(wghts,rets.T)
            
            #append to final array
            mvars.append(oosrets)

    return mvars

def _portfolio_turnover_rate(weights_df):
    turns = list()
    for j in range(1,weights_df.shape[1]):
        turn = np.abs(weights_df[j]-weights_df[j-1])
        turns.append(np.sum(turn))
    
    return round(np.mean(turns),4)


# In[149]:


def _performance_results(returns_df, index_oos_df, optimized_weights_df, n_rebalances):
    mvar_returns = _portfolio_meanvar_oosample_returns(optimized_weights_df, returns_df, n_rebalances)

    #Sharpe ratio
    mvar_sharpe = np.mean(mvar_returns)/np.std(mvar_returns)  #risk free rate = 0
    print("Annualized Sharpe Ratio: ",round(mvar_sharpe*np.sqrt(52),4))

    #Avg return
    mvar_avgreturns = np.mean(mvar_returns)
    print("Annualized Avg Returns: ",round(mvar_avgreturns*52,4))

    #Sortino ratio
    if np.std(np.where(np.array(mvar_returns) > 0, 0, mvar_returns)) != 0:
        mvar_sortino = np.mean(mvar_returns) / np.std(np.where(np.array(mvar_returns) > 0, 0, mvar_returns))
        print('Annualized Sortino Ratio: ',round(mvar_sortino*np.sqrt(52),4))

    #Information ratio
    mvar_ir = float(np.mean(pd.DataFrame(mvar_returns) - index_oos_df) / np.std(pd.DataFrame(mvar_returns) - index_oos_df))
    print('Annualized Information Ratio: ',round(mvar_ir*np.sqrt(52),4))

    #Jensen alpha
    oos_mv1 = np.concatenate((pd.DataFrame(mvar_returns),index_oos_df),axis=1)
    ab = pd.DataFrame(oos_mv1).cov()[0][1]
    mvar_jalpha = float(np.mean(pd.DataFrame(mvar_returns))-(ab / np.var(index_oos_df)) * np.mean(index_oos_df))
    print('Annualized Jensen Alpha: ',round(mvar_jalpha*52,4))

    #Omega ratio
    a = pd.DataFrame(mvar_returns) - np.mean(index_oos_df)
    if np.mean(np.where(a > 0, 0, a)) != 0:
        mvar_omega = float(np.mean(np.where(a < 0, 0, a)) / -np.mean(np.where(a > 0, 0, a)))
        print('Annualized Omega Ratio: ',round(mvar_omega*52,4))

    #Excess mean return
    print("Annualized Excess Mean Returns: ",round(float(np.mean(pd.DataFrame(mvar_returns) - index_oos_df)*52),4))

    #Turnover rate
    print("Turnover Rate: ",_portfolio_turnover_rate(optimized_weights_df))

    return


# In[156]:


print("FTSE100 Mean-Variance")
_performance_results(ftse_ass_df, ftse_oos, ftse_optmv_df, ftse_nrebs)
print("\nDJIA Mean-Variance")
_performance_results(djia_ass_df, djia_oos, djia_optmv_df, djia_nrebs)
print("\nFF49 Mean-Variance")
_performance_results(ff49_ass_df, ff49_oos, ff49_optmv_df, ff49_nrebs)
print("\nS&P500 Mean-Variance")
_performance_results(sp_ass_df, sp_oos, sp_optmv_df, sp_nrebs)
print("\nNASDAQ100 Mean-Variance")
_performance_results(nd100_ass_df, nd100_oos, nd100_optmv_df, nd100_nrebs)


# In[ ]:





# In[ ]:





# ### Mean-Variance optimization using sklearn

# In[159]:


def _mv_performance(arr_weights, mean_returns, covariance_matrix):
    returns = np.sum(mean_returns*arr_weights) #* 52
    risk = np.sqrt(np.dot(arr_weights.T, np.dot(covariance_matrix, arr_weights))) #* np.sqrt(52)
    
    return risk, returns

'''
    Function to return the negative of the sharpe ratio calculation given the weights, the mean of returns array,
    the covariance matrix and the risk free rate.
    Required for minimize optimisation 
'''
def _neg_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate):
    p_std, p_ret = _mv_performance(weights, mean_returns, covariance_matrix)
    return - (p_ret - risk_free_rate) / p_std

'''
    Function that applies scipy minimize optimiser given the mean daily returns array, the covariance matrix
    and the risk free rate. Returns the resulting optimisation result array for the maximum sharpe ration
    using SLSQP method.
'''
def _max_sharpe_ratio(mean_returns, covariance_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(_neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

'''
    Function to return the volatility calculation given the weights, the mean of returns array
    and the covariance matrix. 
    Required for minimize optimisation 
'''
def _volatility(weights, mean_returns, covariance_matrix):
    return _mv_performance(weights, mean_returns, covariance_matrix)[0]

'''
    Function that applies scipy minimize optimiser given the mean daily returns array and the covariance matrix.
    Returns the resulting optimisation result array for the minimum volatility using SLSQP method.
'''
def _min_volatility(mean_returns, covariance_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# In[46]:


#ftse_ass_df.iloc[0:52,:].cov()


# In[47]:


def _maxsharpe_opt_weights(asset_returns_df, n_rebalances):
    insample_start = 0
    rolling_length = 12
    insample_length = 52
    oosample_start = 52
    oosample_length = 12

    weights_df = pd.DataFrame()

    for j in range(n_rebalances):
        #move rolling window parameters
        insample_start = j*rolling_length
        oosample_start = insample_length + (j*rolling_length)

        max_return = _max_sharpe_ratio(asset_returns_df.iloc[insample_start:oosample_start-1,:].mean(), 
                                       asset_returns_df.iloc[insample_start:oosample_start-1,:].cov(), 0)
        weights_df = pd.concat([weights_df, pd.DataFrame(max_return.x)], axis=1, ignore_index=True)

    return weights_df

def _minvol_opt_weights(asset_returns_df, n_rebalances):
    insample_start = 0
    rolling_length = 12
    insample_length = 52
    oosample_start = 52
    oosample_length = 12

    weights_df = pd.DataFrame()

    for j in range(n_rebalances):
        #move rolling window parameters
        insample_start = j*rolling_length
        oosample_start = insample_length + (j*rolling_length)

        minvol_return = _min_volatility(asset_returns_df.iloc[insample_start:oosample_start-1,:].mean(),
                                        asset_returns_df.iloc[insample_start:oosample_start-1,:].cov())
        weights_df = pd.concat([weights_df, pd.DataFrame(minvol_return.x)], axis=1, ignore_index=True)

    return weights_df


# In[50]:





# In[158]:


print("FTSE100 Mean-Variance - Maximum Sharpe Ratio")
ftse_opt_weights_df = _maxsharpe_opt_weights(ftse_ass_df, ftse_nrebs)
_performance_results(ftse_ass_df, ftse_oos, ftse_opt_weights_df, ftse_nrebs)

print("\nFTSE100 Mean-Variance - Minimum Volatility")
ftse_opt_weights_df = _minvol_opt_weights(ftse_ass_df, ftse_nrebs)
_performance_results(ftse_ass_df, ftse_oos, ftse_opt_weights_df, ftse_nrebs)


# In[160]:


print("DJIA Mean-Variance - Maximum Sharpe Ratio")
djia_opt_weights_df = _maxsharpe_opt_weights(djia_ass_df, djia_nrebs)
_performance_results(djia_ass_df, djia_oos, djia_opt_weights_df, djia_nrebs)

print("\nDJIA Mean-Variance - Minimum Volatility")
djia_opt_weights_df = _minvol_opt_weights(djia_ass_df, djia_nrebs)
_performance_results(djia_ass_df, djia_oos, djia_opt_weights_df, djia_nrebs)


# In[161]:


print("FF49 Mean-Variance - Maximum Sharpe Ratio")
ff49_opt_weights_df = _maxsharpe_opt_weights(ff49_ass_df, ff49_nrebs)
_performance_results(ff49_ass_df, ff49_oos, ff49_opt_weights_df, ff49_nrebs)

print("\nFF49 Mean-Variance - Minimum Volatility")
ff49_opt_weights_df = _minvol_opt_weights(ff49_ass_df, ff49_nrebs)
_performance_results(ff49_ass_df, ff49_oos, ff49_opt_weights_df, ff49_nrebs)


# In[162]:


print("S&P500 Mean-Variance - Maximum Sharpe Ratio")
sp_opt_weights_df = _maxsharpe_opt_weights(sp_ass_df, sp_nrebs)
_performance_results(sp_ass_df, sp_oos, sp_opt_weights_df, sp_nrebs)

print("\nS&P500 Mean-Variance - Minimum Volatility")
sp_opt_weights_df = _minvol_opt_weights(sp_ass_df, sp_nrebs)
_performance_results(sp_ass_df, sp_oos, sp_opt_weights_df, sp_nrebs)


# In[165]:


print("NASDAQ100 Mean-Variance - Maximum Sharpe Ratio")
nd100_opt_weights_df = _maxsharpe_opt_weights(nd100_ass_df, nd100_nrebs)
_performance_results(nd100_ass_df, nd100_oos, nd100_opt_weights_df, nd100_nrebs)

print("\nNASDAQ100 Mean-Variance - Minimum Volatility")
nd100_opt_weights_df = _minvol_opt_weights(nd100_ass_df, nd100_nrebs)
_performance_results(nd100_ass_df, nd100_oos, nd100_opt_weights_df, nd100_nrebs)


# In[ ]:





# In[ ]:





# In[ ]:





# ### SVR for returns prediction

# In[117]:


#Normalization
def _normalize(df):

    # Get column names first
    names = df.columns

    # Create the Scaler object based on range [-1,1]
    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    #scaled_ftse_ass_df.head()

    #test invere tansform
    #test_invscale_df = pd.DataFrame(scaler.inverse_transform(scaled_ftse_ass_df), columns=names)
    #test_invscale_df

    scaled_df = scaled_df.reset_index()
    scaled_df.rename(columns={'index':'week'}, inplace=True)
    #scaled_df.head()

    return scaler,scaled_df


# In[53]:


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


# ### RF for returns prediction

# In[54]:



def predict_returns_RF(dates, returns, x, n_est_coef, rnd_coef):
    # convert to 1xn dimension
    dates = np.reshape(dates,(len(dates), 1)) 
    x_test = np.reshape(x,(len(x), 1))
    #print(dates)
    #print(returns)
    
    rf = RandomForestRegressor(n_estimators=n_est_coef, random_state=rnd_coef, criterion='mse', max_depth=5)   #n_estimators = 1000, random_state = 42
    
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


# In[55]:


#print(scaled_ftse_ass_df.head())
#scaled_ftse_ass_df[1][1]
#print(scaled_ftse_ass_df.iloc[0:5,1].values)

#pred_returns_rbf = scaled_ftse_ass_df.copy()
#pred_returns_rbf.iloc[:,1:]=0.0
#pred_returns_rbf.iloc[0:5,1] = scaled_ftse_ass_df.iloc[0:5,1].values
#print(pred_returns_rbf.head(10))


# In[106]:


# Find predicted returns (main)

def predict_main(df, use_expost, insample_start, oosample_start, insample_length, oosample_length, rolling_length, 
                 nstocks, nrebs, use_SVR=True, SVR_C=1000, SVR_gamma=0.1, use_RF=False, RF_nest=0, RF_rand=42):

    #copy dataframe and initialise values
    pred_returns_rbf = df.copy()
    pred_returns_rbf.iloc[:,1:] = 0.0
    #print(pred_returns_rbf.head())

    for nn in range(nstocks):
        ins_st = insample_start
        oos_st = oosample_start
        ins_len = insample_length
        oos_len = oosample_length
        roll_len = rolling_length

        print("=========")
        print('Stock ',nn)
        #print("=========")

        #initialize in-sample array
        insample_returns_rbf = df[nn]

        pred_returns_rbf.iloc[0:oos_st,nn+1] = df.iloc[0:oos_st,nn+1].values
        #print(df.iloc[0:oos_st,nn+1].values)
        #print(pred_returns_rbf.head(60))

        for nwin in range(nrebs):
            #print("     ~~~~~~~~~~")
            print('     Window ',nwin)
            #print("     ~~~~~~~~~~")

            #advance insample and o-o-sample start positions by rolling length
            if nwin > 0:
                ins_st = nwin*roll_len
                oos_st = ins_len + (nwin*roll_len)

            if nwin == nrebs-1:
                oos_len = (df.shape[0] - ins_len)%roll_len
                
            for i in range(oos_len):
                #print('          Iteration ',i+1) #, ins_st, oos_st)

                #set in-sample data
                weeks = df['week'][i+ins_st:i+oos_st,].tolist()
                #print(i+ins_st,i+oos_st,len(weeks),weeks)

                if use_expost == True:
                    #ex-post (actual) returns
                    insample_returns_p = df[nn][i+ins_st:i+oos_st,].tolist() 
                    #print(scaled_ftse_ass_df[nn][i+ins_st:i+oos_st,].tolist())
                    #print(len(insample_returns_p),insample_returns_p)

                    #predict one-step ahead return using ex-post (actual) returns
                    if use_SVR == True:
                        outofsample_predicted_returns = predict_returns_SVR(weeks, insample_returns_p, [i+oos_st], 
                                                                            SVR_C, SVR_gamma)
                    else:
                        outofsample_predicted_returns = predict_returns_RF(weeks, insample_returns_p, [i+oos_st], 
                                                                           RF_nest, RF_rand)
                    
                    #print('Predicted returns for week:',i+oos_st)
                    #print(len(outofsample_predicted_returns),outofsample_predicted_returns)
                else:
                    #ex-ante (predicted) returns
                    insample_returns = insample_returns_rbf[i+ins_st:i+oos_st,].tolist() 
                    #print(insample_returns_rbf[ins_st:i+ins_st,].tolist())

                    #predict one-step ahead return using ex-ante (predicted) returns
                    if use_SVR == True:
                        outofsample_predicted_returns = predict_returns_SVR(weeks, insample_returns, [i+oos_st], 
                                                                            SVR_C, SVR_gamma)
                                                        #scaled_ftse_ass_df[nn][i+oos_st].tolist()) 
                    else:
                        outofsample_predicted_returns = predict_returns_RF(weeks, insample_returns, [i+oos_st], 
                                                                           RF_nest, RF_rand)
                    
                #add result to predicted returns dataframe
                pred_returns_rbf.iloc[i+oos_st,nn+1] = outofsample_predicted_returns[0]
                #print(pred_returns_rbf.head(60))


                #insample_returns_lin = np.concatenate((np.array(insample_returns),outofsample_predicted_returns[0]),axis=0)
                #add predicted return to in-sample for the next iteration
                insample_returns_rbf = np.concatenate((np.array(insample_returns_rbf[0:i+oos_st,]),
                                                       outofsample_predicted_returns),axis=0)

                #print('Rolling returns with SVR(rbf):')
                #print(len(insample_returns_rbf),insample_returns_rbf)

    return pred_returns_rbf


# In[110]:


# Set training parameters
insample_start = 0
oosample_start = 52
insample_length = 52
oosample_length = 12
rolling_length = 12

# For FTSE100
ftse_scaler, scaled_ftse_ass_df = _normalize(ftse_ass_df)

# For DJIA
#nstocks = 2
#nrebs = 110


# In[124]:


# Run time approx 40 mins

#C=1000
#gamma=0.1
returns_svr_rbf1 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                               insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
                               use_SVR=True, SVR_C=1000, SVR_gamma=0.1, use_RF=False)
returns_svr_rbf1

#C=1000
#gamma=0.3
#returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                               insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
#                               use_SVR=True, SVR_C=1000, SVR_gamma=0.3, use_RF=False)
#returns_svr_rbf2

#C=1000
#gamma=0.5
#returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                               insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
#                               use_SVR=True, SVR_C=1000, SVR_gamma=0.5, use_RF=False)
#returns_svr_rbf3

#C=10000
#gamma=0.2
#returns_svr_rbf = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                               insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
#                               use_SVR=True, SVR_C=10000, SVR_gamma=0.2, use_RF=False)
#returns_svr_rbf4


# In[126]:


returns_svr_rbf1.drop(columns=['week'],inplace=True)

#Descaled returns from SVR
ftse_svr_df1 = pd.DataFrame(ftse_scaler.inverse_transform(returns_svr_rbf1), columns=names)
ftse_svr_df1.head(20)


# In[166]:


print("FTSE100 Mean-Variance - Maximum Sharpe Ratio (SVR)")
ftse_opt_weights_df = _maxsharpe_opt_weights(ftse_svr_df1, ftse_nrebs)
_performance_results(ftse_svr_df1, ftse_oos, ftse_opt_weights_df, ftse_nrebs)


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


# In[171]:


#n_est=100
ftse_rf_df1 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
                           insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
                           use_SVR=False, use_RF=True, RF_nest=100)
ftse_rf_df1

#n_est=300
#returns_rf2 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                           insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
#                           use_SVR=False, use_RF=True, RF_nest=50)
#returns_svr_rbf2

#n_est=500
#returns_rf3 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                           insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
#                           use_SVR=False, use_RF=True, RF_nest=500)
#returns_svr_rbf3

#n_est=1000
#returns_rf4 = predict_main(scaled_ftse_ass_df, True, insample_start, oosample_start, 
#                           insample_length, oosample_length, rolling_length, ftse_nstocks, ftse_nrebs,
#                           use_SVR=False, use_RF=True, RF_nest=1000)
#returns_svr_rbf4


# In[172]:


ftse_rf_df1.drop(columns=['week'],inplace=True)

#Descaled returns from SVR
ftse_rf_df11 = pd.DataFrame(scaler.inverse_transform(ftse_rf_df1), columns=names)
ftse_rf_df11.head(20)


# In[173]:


#Use ex-ante (predicted) returns with mean variance
ftse_opt_weights_df = _maxsharpe_opt_weights(ftse_rf_df11, ftse_nrebs)
_performance_results(ftse_rf_df11, ftse_oos, ftse_opt_weights_df, ftse_nrebs)


# In[ ]:





# In[ ]:





# In[251]:



def predict_error(model, act_df, pred_df, nstocks, start_index, end_index):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
        mape = mean_absolute_percentage_error(y_actual, y_predicted)
        print('MAPE ',mape)
        mdae = median_absolute_error(y_actual, y_predicted)
        print('MedAE ',mdae)
        #print('Confusion matrix:')
        #cm = confusion_matrix(y_actual,y_predicted)
        #print(cm)
    return rms,mae,mdae


# In[102]:


# Plot actual vs. predicted returns data for 1st stock
#plt.scatter(dates, returns, c='k', label='Data')
plt.plot(scaled_ftse_ass_df['week'][52:711], scaled_ftse_ass_df[0][52:711], c='g', label='actual')
#plt.plot(scaled_ftse_ass_df['week'][52:711], returns_svm_rbf[0][52:711], c='r', label='predicted')    
#plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Actual returns normalized [-1,1]')
plt.legend()
plt.show()

plt.plot(scaled_ftse_ass_df['week'][52:711], returns_svr_rbf1[0][52:711], c='r', label='predicted')    
#plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Support Vector Regression predicted returns')
plt.legend()
plt.show()


# In[253]:


#measure prediction error
predict_error('SVR(rbf)', scaled_ftse_ass_df, returns_svr_rbf1, nstocks, oosample_start, 711)
#predict_error('SVR(rbf)2', scaled_ftse_ass_df, returns_svr_rbf2, nstocks, 52, 711)
#predict_error('SVR(rbf)3', scaled_ftse_ass_df, returns_svr_rbf3, nstocks, 52, 711)
#predict_error('SVR(rbf)4', scaled_ftse_ass_df, returns_svr_rbf4, nstocks, 52, 711)

#predict_error('SVR(rbf)5', scaled_ftse_ass_df, returns_svr_rbf5, nstocks, 52, 711)
#predict_error('SVR(rbf)6', scaled_ftse_ass_df, returns_svr_rbf6, nstocks, 52, 711)
#predict_error('SVR(rbf)7', scaled_ftse_ass_df, returns_svr_rbf7, nstocks, 52, 711)
#predict_error('SVR(rbf)8', scaled_ftse_ass_df, returns_svr_rbf8, nstocks, 52, 711)
#predict_error('SVR(rbf)9', scaled_ftse_ass_df, returns_svr_rbf9, nstocks, 52, 711)



# In[252]:


predict_error('RF', scaled_ftse_ass_df, returns_rf1, nstocks, oosample_start, 711)
#predict_error('RF2', scaled_ftse_ass_df, returns_rf2, nstocks, oosample_start, 711)
#predict_error('RF3', scaled_ftse_ass_df, returns_rf3, nstocks, oosample_start, 711)
#predict_error('RF4', scaled_ftse_ass_df, returns_rf4, nstocks, oosample_start, 711)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[305]:


#ftse_ass_df


# In[301]:





# In[ ]:


# Efficient Frontier


# In[ ]:





# In[708]:


#NASDAQComp
#ndcom_meanreturns = ndcom_ass_df.mean()
#ndcom_covreturns = ndcom_ass_df.cov()
#ndcom_covreturns
#in_sample_results1(ndcom_ass_df,'NASDAQComp',ndcom_meanreturns,ndcom_covreturns,0)


# In[ ]:





# In[ ]:





# In[503]:



#np.random.seed(42)
#numPortfolios = 100
#riskFreeRate = 0
#results, asset_weights = _random_portfolios(numPortfolios, meanWklyreturns, 
#                                            covreturns, riskFreeRate)
#
#print(results)
#print(results.shape)
#print(ftse_ass_df.shape[1])
#portfolio = {'Returns': results[1], 'Volatility': results[0], 'Sharpe Ratio': results[2]}
#
# extend original dictionary to accomodate each ticker and weight in the portfolio
#for counter in range(ftse_ass_df.shape[1]):   #,symbol in enumerate(str(ftse_ass_df.columns-1)):
#    portfolio[str(counter)+' Weight'] = [Weight[counter] for Weight in asset_weights]
#
# make a nice dataframe of the extended dictionary
#df = pd.DataFrame(portfolio)
#
# get better labels for desired arrangement of columns
#column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [str(stock)+' Weight' for stock in ftse_ass_df.columns]
#print(column_order)
#
# reorder dataframe columns
#df = df[column_order]
#-----
#
#df

#plt.figure(figsize=(10, 7))
#plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='twilight_shifted', 
#            marker='o', s=10, alpha=0.5)
#plt.colorbar()
#plt.title('Calculated Portfolio Optimization based on 2000 random portfolios')
#plt.xlabel('Annualised Volatility')
#plt.ylabel('Annualised Returns')
#
#plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
#plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum Volatility')
#
#plt.title('Calculated Portfolio Optimization based on 2000 random portfolios')
#plt.xlabel('Annualised Volatility')
#plt.ylabel('Annualised Returns')
#plt.legend(labelspacing=0.8)
#
#target = np.linspace(rp_min, 0.005, 10)
#efficient_portfolios = _efficient_frontier(meanWklyreturns, covreturns, target)
#plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', 
#         label='Efficient Frontier')


# In[ ]:





# In[526]:


import cvxpy as cvx
 
def proj_simplex_cvxpy(a, y, gammapref, rhopref):
    '''
    Returns the point in the simplex a^Tx = 1, x>=0 that is
     closest to y (according to Euclidian distance)
    '''
    d = len(a)
 
    # setup the objective and constraints and solve the problem
    x = cvx.Variable(d)
    obj = cvx.Maximize((x - gammaprref)*(rhopref - y))   #Minimize(cvx.sum_squares(x - y))
    constr = [x >= gammapref, y <= rhopref] #a*x == 1]
    prob = cvx.Problem(obj, constr)
    prob.solve()
 
    return np.array(x.value).squeeze()


# In[ ]:


proj_simplex_cvxpy(ftse_ass_df,)


# In[ ]:





# In[ ]:





# In[719]:


import pypfopt.efficient_frontier as ppfo
from pypfopt import risk_models
from pypfopt import expected_returns

#mu = ftse_ass_df.mean() #expected_returns.mean_historical_return(ftse_ass_df)
#mu
df11 = ftse_ass_df.iloc[0:52]
#print(df11)
ef = ppfo.EfficientFrontier(df11.mean(),covreturns,(0,1))
raw_weights = ef.max_sharpe(risk_free_rate=0) #min_volatility() #
print(raw_weights)
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

#ef.portfolio_performance(verbose=True,risk_free_rate=0)
#ef.efficient_return(0.05)


# In[ ]:




