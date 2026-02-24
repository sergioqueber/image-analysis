# Exercise 1 

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

in_dir = 'data/'
txt_name = 'irisdata.txt'
iris_data = np.loadtxt(in_dir + txt_name, comments="%")

x = iris_data[0:50, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]
print ('Number of features: %d' % n_feat)
print ('Number of observations: %d' % n_obs)

#Exercise 2

sep_1 = x[:,0]
sep_w = x[:,1]
pet_1 = x[:,2]
pet_w = x[:,3]

var_sep_1 = sep_1.var(ddof = 1)
var_sep_w = sep_w.var(ddof = 1)
var_pet_1 = pet_1.var(ddof = 1)
var_pet_w = pet_w.var(ddof = 1)

# We use ddof = 1 to get the sample variance, which divides by (n-1) instead of n. This is important when we have a small sample size, as it provides an unbiased estimate of the population variance.

print ('Variance of sepal length: %f' % var_sep_1)
print ('Variance of sepal width: %f' % var_sep_w)
print ('Variance of petal length: %f' % var_pet_1)
print ('Variance of petal width: %f' % var_pet_w)

# Exercise 3 Compute the covariance between the sepal length and the petal length and compare it to the covariance between the sepal length and width. What do you observe?

# Manual covariance calculation using the formula: cov = (1/(N-1)) * sum(a_i * b_i)
# First, we need to center the data (subtract the mean from each variable)

# Center sepal length
sep_1_centered = sep_1 - sep_1.mean()

# Center petal length
pet_1_centered = pet_1 - pet_1.mean()

# Center sepal width
sep_w_centered = sep_w - sep_w.mean()

# Compute covariance between sepal length and petal length using the formula
N = len(sep_1)
cov_sep1_pet1 = (1 / (N - 1)) * np.sum(sep_1_centered * pet_1_centered)

# Compute covariance between sepal length and sepal width using the formula
cov_sep1_sepw = (1 / (N - 1)) * np.sum(sep_1_centered * sep_w_centered)

print('Manual covariance between sepal length and petal length: %f' % cov_sep1_pet1)
print('Manual covariance between sepal length and sepal width: %f' % cov_sep1_sepw)

# Verify with numpy's built-in function
cov_sep1_pet1_builtin = np.cov(sep_1, pet_1, ddof=1)[0, 1]
cov_sep1_sepw_builtin = np.cov(sep_1, sep_w, ddof=1)[0, 1]

print('\nVerification with numpy.cov:')
print('Built-in covariance between sepal length and petal length: %f' % cov_sep1_pet1_builtin)
print('Built-in covariance between sepal length and sepal width: %f' % cov_sep1_sepw_builtin)

# Exercise 4 

plt.figure()

d = pd.DataFrame(x, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])

sns.pairplot(d)
plt.show()

# What measurements are related and which ones are not-related? Can you recognise the results you found, when you computed the variance and covariance?

# We caan see the strong correlation between sepal length and sepal width 

#Exercise 5 

mn = np.mean(x, axis=0)

data = x - mn
print('Data after centering: %s' % data)

N = data.shape[0]
cov_matrix = (1 / (N - 1)) * np.matmul(data.T, data)

print('Covariance matrix:\n%s' % cov_matrix)

#Verify with numpy's built-in function
cov_matrix_builtin = np.cov(data, rowvar=False, ddof=1)
print('\nVerification with numpy.cov:\n%s' % cov_matrix_builtin)

#Exercise 6 
#compute the principal components using eigenvector analysis

values, vectors = np.linalg.eig(cov_matrix)

print('Eigenvalues:\n%s' % values)
print('\nEigenvectors:\n%s' % vectors)

#Exercise 7 

v_norm = values / values.sum() *100

plt.plot(v_norm, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.ylim([0, 100])
plt.title('Scree Plot')
plt.show()

# Exercise 8

# Projection of data into PCA space using dot product 

pc_proj = vectors.T.dot(data.T)

#Using pairplot 

pc_df = pd.DataFrame(pc_proj.T, columns=['PC1', 'PC2', 'PC3', 'PC4'])
sns.pairplot(pc_df)
plt.suptitle('Pairplot of PCA-projected data', y=1.02)
plt.show()

#Exercise 9 

pca  = decomposition .PCA()
pca.fit(x)

values_pca = pca.explained_variance_
exp_var_ratio_pca = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)

plt.figure()
d = pd.DataFrame(data_transform, columns=['PC1', 'PC2', 'PC3', 'PC4'])

p = sns.pairplot(d)
p.set(xlim=(-1,1), ylim = (-1,1))
plt.show()

print('PCA explained variance:\n%s' % values_pca)
print('\nPCA explained variance ratio:\n%s' % exp_var_ratio_pca)
print('\nPCA components:\n%s' % vectors_pca)

##We can see that the eigen vector are transposed with regards to the vectors from the pca, which is due to the fact that the PCA components are stored as rows in the PCA object, while in our manual calculation, we stored them as columns. The explained variance and explained variance ratio from PCA should match the eigenvalues and their normalized version from our manual calculation, confirming that both methods yield the same results.


