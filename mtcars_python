#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import statsmodels.api as sm


# In[111]:


from statsmodels.formula.api import ols
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[4]:


mtcars = pd.read_csv('https://gist.githubusercontent.com/ZeccaLehn/4e06d2575eb9589dbe8c365d61cb056c/raw/898a40b035f7c951579041aecbfb2149331fa9f6/mtcars.csv')


# In[12]:


pd.DataFrame(mtcars).head()


# In[17]:


model = ols('mpg~cyl+disp+hp+drat+wt+qsec+vs+am+gear+carb', data = mtcars).fit()
print(model.summary())


# In[18]:


model = ols('mpg~disp+hp+drat+wt+qsec+vs+am+gear+carb', data = mtcars).fit()
print(model.summary())


# In[19]:


model = ols('mpg~disp+hp+drat+wt+qsec+am+gear+carb', data = mtcars).fit()
print(model.summary())


# In[20]:


model = ols('mpg~disp+hp+drat+wt+qsec+am+gear', data = mtcars).fit()
print(model.summary())


# In[21]:


model = ols('mpg~disp+hp+drat+wt+qsec+am', data = mtcars).fit()
print(model.summary())


# In[22]:


model = ols('mpg~disp+hp+wt+qsec+am', data = mtcars).fit()
print(model.summary())


# In[23]:


model = ols('mpg~hp+wt+qsec+am', data = mtcars).fit()
print(model.summary())


# In[24]:


model = ols('mpg~wt+qsec+am', data = mtcars).fit()
print(model.summary())


# In[64]:


fitted = model.predict()
fitted


# In[96]:


residual = model.resid


# In[97]:


plt.scatter(fitted, residual)
plt.title("fitted vs residual")
plt.xlabel("fitted")
plt.ylabel("residual")
plt.show()


# In[110]:


shapiro(residual)


# In[122]:


x_model = mtcars[['wt', 'qsec', 'am']]
x_new = sm.add_constant(x_model)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = x_new.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x_new.values, i)
                          for i in range(len(x_new.columns))]
  
print(vif_data[1:])

