import pandas as pd  
import numpy as np  
from sklearn import datasets

iris = datasets.load_iris()
X_train = iris.data[:,:]
ff=iris.DESCR
print(ff)
iris = pd.DataFrame(X_train)



""" Alldata="/var/home/nhamad/k-means/dataset4.csv"
TRdata = pd.read_csv(Alldata, encoding='utf-8') 
pd = pd.DataFrame(TRdata) """


print(pd.describe())

""" a1 = pd.Series(['p', 'q', 'q', 'r'])  
a1.describe()  
info = pd.DataFrame({'categorical': pd.Categorical(['s','t','u']),  
'numeric': [1, 2, 3],  
'object': ['p', 'q', 'r']  
 })  
info.describe()  
info.describe(include='all')  
info.numeric.describe()  
info.describe(include=[np.number])  
info.describe(include=[np.object])  
info.describe(include=['category'])  
info.describe(exclude=[np.number])  
info.describe(exclude=[np.object])   """