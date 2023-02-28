import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
from sklearn import datasets
import math 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def euclidean(point, cent):

    minus=np.subtract(point, cent)
    ss=np.square(minus)
    sumxx = np.sum(ss,axis=1)
    dist=np.sqrt(sumxx)            
    return dist
    #return np.sqrt(np.sum((point - data)**2, axis=1))
    
def modify_U_matrix(num,centroid_idx,u):
    u[centroid_idx][num]=1
    return u

class KMeans:
    def __init__(self, n_clusters=2, max_iter=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        
    def fit(self, X_train):
        # Initialize the centroids, 
    
        self.centroids=[]
        lenn=len(X_train)
        randomRows = np.random.randint(lenn-1, size=self.n_clusters)
        for i in randomRows:
            self.centroids.append(X_train[i, :])
        
        self.centroids=np.asarray(self.centroids)
        
        #print(self.centroids)
        # here must add the length of thr data train
        cent_len=len(self.centroids)
        x_train_len=len(X_train)
    
          
        i=0
        self.iteration = 0
        self.prev_centroids = None
        threshold=0.001
        cost_func=0.1
        all_dists=[]
        error=0
        J=0
        th=1
        self.J_values=[]
        #or self.iteration < self.max_iter
        while th>threshold :
            num=0 # this is to count the number of data
            all_dists=[]
            
            self.point_in_cluster_index=[]
            #create a u array with zeros with size numof centroids x number of points in the training dataset
            u= [np.zeros(x_train_len)]
            i=0
            while i < cent_len-1:   
                u.append(np.zeros(x_train_len))
                i+=1
            for x in X_train:
                dists = euclidean(x,self.centroids)
                
                #return the index of lowest number of centroid
                centroid_idx = np.argmin(dists)
                all_dists.append(dists[centroid_idx]) #calculate all distances between each point and centroids
                self.point_in_cluster_index.append(centroid_idx) # append each point in which cluster
                # assign the index to the U matrix
                u=modify_U_matrix(num,centroid_idx,u)
                num=num+1
                #sorted_points[centroid_idx].append(x)
            
            # now find the mean for each clusteros
            
            new_centroids=[]
            for x in u:
                ind=np.where(x == 1)[0]
                
                # now calculate 
                points=[]
                for y in ind:
                    points.append(X_train[y][:])
                       
                mean_cen=np.mean(points, axis=0)
                if(np.isnan(mean_cen).any()):
                    continue
                new_centroids.append(mean_cen)
            self.centroids=np.asarray(new_centroids)
            
            #calculate the distance for each point in each cluster
            
            
            dd=[]
            i=0
            allsum=0
            for x in u:
                ind=np.where(x == 1)[0]        
                # now calculate 

                dist=euclidean(X_train[ind][:],self.centroids[i])
                #sub=((X_train[y][:])-(self.centroids[i]))**2
                sum_each_cluster = np.sum(dist,axis=0) 
                #dd.append(sum_each_cluster)
                allsum+=sum_each_cluster
                i+=1
                   
            J=allsum/lenn
            
            self.J_values.append(J)
            if(self.iteration >=1):
                th=(self.J_values[self.iteration-1]- self.J_values[self.iteration]) / self.J_values[self.iteration-1]

            
            total_cost = (all_dists[-1]**2).sum()
            error=(total_cost-cost_func)/cost_func
            self.iteration += 1
        df = pd.DataFrame(self.centroids)

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs


# 1- read from dataset
centers = 5
""" iris = datasets.load_iris()
X_train = iris.data[:,:]# take the features
true_labels = iris.target #take the label  """

Alldata="/var/home/nhamad/k-means/dataset3.csv"
TRdata = pd.read_csv(Alldata, encoding='utf-8') 
X_train = pd.DataFrame(TRdata, columns = ['f1','f2'])
true_labels = pd.DataFrame(TRdata, columns = ['label'])

X_train = X_train.to_numpy()
true_labels=true_labels.to_numpy().flatten()

# 2- normalize the dataset
X_train = StandardScaler().fit_transform(X_train)# normilzation


kmeans = KMeans(n_clusters=centers,max_iter=500)

kmeans.fit(X_train)
# View results

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
points = pipeline.fit_transform(X_train)
points = pd.DataFrame(points)
x = points[0]
y = points[1]
#print("centt")
#print(c)

c = pipeline.fit_transform(kmeans.centroids)
c = pd.DataFrame(c)
centroids_x=c[0]
centroids_y= c[1]

#plt.plot(np.arange(len(L)),L)
""" plt.scatter(x, y, c=kmeans.point_in_cluster_index)
plt.plot(centroids_x, centroids_y, c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=22)
plt.savefig('testmoon2.png') """

#class_centers, classification = kmeans.evaluate(X_train)
figure(figsize=(11, 5), dpi=300)

plt.subplot(1, 2, 1)
sns.scatterplot(x,
                y,
                hue=kmeans.point_in_cluster_index,
                style=kmeans.point_in_cluster_index,
                palette="deep",
                legend=None
                )

plt.plot(centroids_x,
         centroids_y,
         'k+',
         markersize=10,
         )                

plt.title("K-means for dataset3")

""" plt.xlabel("sepal_length")
plt.ylabel("petal_length")
plt.savefig('dataset43.png')
plt.show() """


plt.subplot(1, 2, 2)
y = np.array(kmeans.J_values)
#y = np.sort(x)
x=np.arange(0, kmeans.iteration)
plt.title("data loss function")
plt.ylabel("Loss-values")
plt.xlabel("number of iteration")
plt.plot(x, y, color="red")

plt.savefig('/var/home/nhamad/k-means/images/dataset3/k_5.png')

        
