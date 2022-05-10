
import pandas as pd 
df = pd.read_csv('wine.csv')
#print(df.head())  


import numpy as np

# how to change a column name
df.columns = df.columns.str.replace('color', 'is_red')

#how to change a value in the DataFrame
df ['is_red'] = np.where(df['is_red'] == 'red', 1, 0) 
 

#new DataFrame
df.drop('quality', inplace=True, axis=1) 
numeric_data = df

#all the red wines
count = 0
#how to iterate a column
for color in numeric_data['is_red']:
    if color == 1:
        count = count + 1




#scaling the data
import sklearn.preprocessing
scaled_data = pd.DataFrame(sklearn.preprocessing.scale(numeric_data), columns=(numeric_data.columns))
numeric_data = scaled_data

#extracting the first two principal components
import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data)
principal_numeric_data = pd.DataFrame(data=principal_components)





import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])

# saving the coordinates
x = principal_components[:,0]
y = principal_components[:,1] 

# creating the plot
plt.title('Principal Components of Wine')
plt.scatter(x, y, alpha = 0.2,
    c =df['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
#plt.show()




#np.random.seed(1)
predictions = np.random.randint(0,2,1000)
outcomes = np.random.randint(0 ,2, 1000)

# the 'accuracy' function
def accuracy(predictions, outcomes):
    counting = 0
    total_len = len(predictions) + len(outcomes)
    for co,ou in zip(predictions,outcomes):
        if co == ou:
            counting = counting + 1
    perc = (counting*2/total_len)*100
    return perc



low_quality = list()
for cr in range(len(df['high_quality'])): 
    low_quality.append(0)

#print(accuracy(low_quality, df['high_quality'])) 



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, df['high_quality'])
library_predictions = knn.predict(numeric_data)
#print(accuracy(library_predictions, df['high_quality']))  



import random 
random.seed(123)
n_rows = df.shape[0]
selection = random.sample(range(n_rows), 10) 
#print(selection)