#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns ; sns.set()


# In[10]:


from sklearn.datasets.samples_generator import make_blobs

X,y = make_blobs(n_samples=50,centers=2,
                random_state = 0,cluster_std =  0.60)
plt.scatter(X[: , 0], X[:, 1], c=y, s=50, cmap='summer');


# In[14]:


xfit = np.linspace(-1, 3.5)
plt.scatter (X[ :, 0] , X[ :, 1] , c=y,s=50, cmap='summer')
plt.plot([0.6],[2.1], 'x' , color='green', markeredgewidth=2, markersize=10 )

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b ,'-k')
plt.xlim(-1,3.5)    


# In[20]:


xfit = np.linspace(-1,3.5)
plt.scatter (X[ :, 0] , X[ :, 1] , c=y,s=50, cmap='summer')

for m, b, d in[(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    y=m * xfit + b
    plt.plot(xfit,y, '-k')
    plt.fill_between(xfit,y - d, y + d ,edgecolor='none',
                   color="#AAAAAA", alpha=0.4)
plt.xlim(-1,3.5);    


# In[25]:


from sklearn.svm import SVC
model = SVC(kernel='linear' , C=1E10)
model.fit(X,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




