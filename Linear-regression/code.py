# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df=pd.read_csv(path)
#print(df.head())
X=df.drop(columns='list_price')
print(X.head())
y=df['list_price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.3,random_state = 6)

# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
cols =X_train.columns
fig ,axes=plt.subplots(3,3,figsize=(10,10))

for i in range(0,3):
    for j in range(0,3):
        
        
        col=cols[(i*3)+j]
        axes[i][j].scatter(X_train[col],y_train)
        axes[i][j].set_xlabel(col)
# code ends here



# --------------
# Code starts here
import numpy as np
corr=X_train.corr()

#print(corr)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
#print(np.triu(np.ones(corr.shape)))
#print(np.ones(corr.shape))
#print(upper)
##print(upper.columns)
##to_drop=[s for s in upper.columns if any(upper[s]>0.75)]
#print('to_drop',to_drop)
to_drop=['play_star_rating','val_star_rating']
X_train.drop(to_drop,axis=1,inplace=True)
X_test.drop(to_drop,axis=1,inplace=True)
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Code starts here
regressor=LinearRegression()
regressor.fit(X_train ,y_train)
y_pred=regressor.predict(X_test)
# Code ends here
mse=mean_squared_error(y_test,y_pred)
print(mse)
r2=round(r2_score(y_test,y_pred),2)
print(r2)


# --------------
# Code starts here

residual =y_test -y_pred

residual
residual.hist()

# Code ends here


