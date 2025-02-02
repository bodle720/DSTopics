# -*- coding: utf-8 -*-
"""
scratch_for_robust_pca
"""
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import tensorflow.keras.datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
# RPCA from Steven L. Brunton; J. Nathan Kutz (2019)
# Available from Ch.3 Sec.7 at http://databookuw.com/CODE_PYTHON.zip

def shrink(X,tau):
    Y = np.abs(X)-tau
    return np.sign(X) * np.maximum(Y,np.zeros_like(Y))
    
def SVT(X,tau):
    U,S,VT = np.linalg.svd(X,full_matrices=0)
    out = U @ np.diag(shrink(S,tau)) @ VT
    return out
    
def RPCA(X):
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    lambd = 1/np.sqrt(np.maximum(n1,n2))
    thresh = 10**(-7) * np.linalg.norm(X)
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X-L-S) > thresh) and (count < 1000):
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
        count += 1
    return L,S


# Specify size of training and testing sets, 10 classes of digits (0 through 9)
n_tr_samples_per_class = 30 # Approximate after random selection.
n_te_samples_per_class = 10 # Approximate after random selection.

# Load in the data from TensorFlow
(x_train, y_train), (x_test, y_test) = tfds.mnist.load_data()

# First, we will reduce the training and testing set to a manageable number
# while maintaining roughly even class splits
sss_tr = StratifiedShuffleSplit(n_splits = 1,
                                train_size = n_tr_samples_per_class*10)

for sample_index, _ in sss_tr.split(x_train, y_train):
    X_tr = x_train[sample_index]
    y_tr = y_train[sample_index]
    
unique_tr_classes, tr_class_counts = np.unique(y_tr, return_counts = True)
tr_class_counts_df = pd.DataFrame({'class':unique_tr_classes, 'counts': tr_class_counts})
print('Training data:\n', tr_class_counts_df)

sss_te = StratifiedShuffleSplit(n_splits = 1,
                                train_size = n_te_samples_per_class*10)

for sample_index, _ in sss_te.split(x_test, y_test):
    X_te = x_test[sample_index]
    y_te = y_test[sample_index]
    
unique_te_classes, te_class_counts = np.unique(y_te, return_counts = True)
te_class_counts_df = pd.DataFrame({'class':unique_te_classes, 'counts': te_class_counts})
print('Testing data:\n', te_class_counts_df)


print('Training data array has shape: ', X_tr.shape)
print('Testing data array has shape: ', X_te.shape)
print('Data is type: ', X_tr.dtype)
print('Data max: ', X_tr.max())
print('Data min: ', X_tr.min())
print('The labels are: ', np.unique(y_tr))

# Plot a sample from each class in the training set.
fig, axs = plt.subplots(2, 5, figsize=(7, 7))
axs = axs.flatten()
plt.suptitle('Handwritten Digits')
for label, ax in zip(np.unique(y_tr), axs):
    ix = np.where(y_tr == label)[0][0]
    ax.set_title(f'Label = {label}')
    ax.imshow(X_tr[ix], cmap = 'gray')
fig.tight_layout()
plt.show()


def corrupt_image(image,
                  salt_perc = 0.07,
                  pepper_perc = 0.07,
                  rect_max_dim = 12):
    
    '''
    This function will add noise to an image in three ways:
        1. Salt and pepper noise.
        2. Gaussian noise.
        3. Block sized pixel removal.
        
    Args:
    - image: The image as a 2-D Numpy array (single band).
    - salt_prob: The percentage of total pixels to be turned to 1.
    - pepper_prob: The percentage of total pixels to be turned to 0.
    - rect_max_dim: The maximum length or width of the randomly deleted regions (pixels set to 0).
    '''
    
    image_copy = np.copy(image)
    total_pixels = image_copy.size

    # With 80% probability, add salt and pepper noise.
    if np.random.uniform() > 0.2:
        # Add salt noise.
        num_salt = np.ceil(salt_perc * total_pixels)
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image_copy.shape]
        image_copy[coords_salt[0], coords_salt[1]] = image_copy.max()
    
        # Add pepper noise.
        num_pepper = np.ceil(pepper_perc * total_pixels)
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_copy.shape]
        image_copy[coords_pepper[0], coords_pepper[1]] = image_copy.min()
        
    # With 80% probability, add gaussian noise. This uses a standard deviation of 10% of the pixel value range.
    if np.random.uniform() > 0.2:
        gaussian_noise = np.random.normal(0, 0.1*(image_copy.max() - image_copy.min()), image_copy.shape)
        image_copy = image_copy + gaussian_noise
        
    # With 80% probability, remove a rectangular block of pixels (set to 0).
    # The height and width will range from 1 to rect_max_dim randomly.
    if np.random.uniform() > 0.2:
        # Pick a starting coordinate for the rectangular block.
        y_upper_left = np.random.randint(0, image_copy.shape[0] - rect_max_dim)
        x_upper_left = np.random.randint(0, image_copy.shape[1] - rect_max_dim)

        # Randomly choose a height and width.
        h, w = np.random.randint(1, rect_max_dim, 2)

        y_end = min(y_upper_left + h, image_copy.shape[0] - 1) # Should be fine, but just in case.
        x_end = min(x_upper_left + w, image_copy.shape[1] - 1)
        
        image_copy[y_upper_left:y_end, x_upper_left:x_end] = image_copy.min()
        
    return image_copy


X_tr_aug = np.array([corrupt_image(img) for img in X_tr])
X_te_aug = np.array([corrupt_image(img) for img in X_te])

# Normalize the data from 0 to 1.
tr_mins = X_tr_aug.min(axis = (1, 2), keepdims=True)
tr_maxs = X_tr_aug.max(axis = (1, 2), keepdims=True)
X_tr_scaled = (X_tr_aug - tr_mins)/(tr_maxs - tr_mins)

te_mins = X_te_aug.min(axis = (1, 2), keepdims=True)
te_maxs = X_te_aug.max(axis = (1, 2), keepdims=True)
X_te_scaled = (X_te_aug - te_mins)/(te_maxs - te_mins)


f, axarr = plt.subplots(3,2)
axarr[0,0].imshow(X_tr[0], cmap = 'gray')
axarr[0,1].imshow(X_tr_scaled[0], cmap = 'gray')
axarr[1,0].imshow(X_tr[1], cmap = 'gray')
axarr[1,1].imshow(X_tr_scaled[1], cmap = 'gray')
axarr[2,0].imshow(X_tr[2], cmap = 'gray')
axarr[2,1].imshow(X_tr_scaled[2], cmap = 'gray')
plt.show()

# This places the images as columns as described above.
X_tr_col = X_tr_scaled.reshape(X_tr_scaled.shape[0], -1).T
X_te_col = X_te_scaled.reshape(X_te_scaled.shape[0], -1).T

# This organizes by increasing class.
ordered_columns_tr = np.argsort(y_tr)
X_tr_org = X_tr_col[:,ordered_columns_tr]

ordered_columns_te = np.argsort(y_te)
X_te_org = X_te_col[:,ordered_columns_te]

y_tr_org = y_tr[ordered_columns_tr]
y_te_org = y_te[ordered_columns_te]
#
X_mean = X_tr_org.mean(axis=1, keepdims=True)
X_centered = X_tr_org - X_mean
L_centered, S = RPCA(X_centered)
L = L_centered + X_mean

# now lets look at cleaned digits

# Pick three indices at random from the training set.
inds = np.random.randint(0, X_tr_org.shape[1] - 1, 5)

f, axarr = plt.subplots(5,2)
for ix, ind in enumerate(inds):
    axarr[ix,0].imshow(X_tr_org[:,ind].reshape((28,28)), cmap = 'gray')
    axarr[ix,1].imshow(L[:,ind].reshape((28,28)), cmap = 'gray')
plt.show()

test_data_centered = X_te_org - X_mean
components = np.linalg.pinv(L_centered) @ test_data_centered
centered_test_results = L_centered @ components
test_results = centered_test_results + X_mean

inds = np.random.randint(0, X_te_org.shape[1] - 1, 5)

f, axarr = plt.subplots(5,2)
for ix, ind in enumerate(inds):
    axarr[ix,0].imshow(X_te_org[:,ind].reshape((28,28)), cmap = 'gray')
    axarr[ix,1].imshow(test_results[:,ind].reshape((28,28)), cmap = 'gray')
plt.show()

#%%
# # Let's also average the digits in L.
L_averaged = np.zeros((L.shape[0], 10)) # 10 classes

# Calculate the average for each group
for group in range(10):
    group_columns = L[:, y_tr_org == group]
    L_averaged[:, group] = group_columns.mean(axis = 1)

# Let's view the averaged digits from L.
f, axarr = plt.subplots(5,2)
for ix in range(10):
    axarr[(ix//2)%5,ix%2].imshow(L_averaged[:,ix].reshape((28,28)), cmap = 'gray')
plt.show()

#%%
# The columns of U will contain the eigendigits.
U, Sigma, Vt = np.linalg.svd(L, full_matrices=False)

plt.figure()
plt.plot(Sigma)
plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Sinular Values')
plt.title('Singular Values on a Log Scale')
plt.grid(True)
plt.show()

number_sing_vals = 50
eigendigits = U[:,:number_sing_vals]

#%% plot the firt 10 eigendigits
f, axarr = plt.subplots(5,2)
for ix in range(10):
    axarr[(ix//2)%5,ix%2].imshow(eigendigits[:,ix].reshape((28,28)), cmap = 'gray')
plt.show()

#%%

# Let's pick a random test image and analyze the loadings.
ind = np.random.randint(0, test_results.shape[1] - 1)

# Find the true label of the image and the image itself.
true_label = y_te_org[ind]
img = test_results[:, ind]

# Project the image onto the eigenbasis.
test_signature = eigendigits.T @ img.reshape(-1, 1)

plt.figure(figsize=(30, 15))  # Optional: set the figure size
plt.bar(np.arange(number_sing_vals), test_signature[:,0])
plt.xlabel('Singular vector', fontsize = 30)
plt.ylabel('Loadings', fontsize = 30)

plt.title(f'Signature for a random test sample, truth = {true_label}', fontsize = 40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()

#%%

# Let's pick a random test image and analyze the loadings.
# ind = np.random.randint(0, test_results.shape[1] - 1)

# Find the true label of the image and the image itself.
true_label = 2
inds = np.where(y_te_org == true_label)[0]

for ind in inds:
    img = test_results[:, ind]
    
    # Project the image onto the eigenbasis.
    test_signature = eigendigits.T @ img.reshape(-1, 1)
    
    plt.figure(figsize=(30, 15))  # Optional: set the figure size
    plt.bar(np.arange(number_sing_vals), test_signature[:,0])
    plt.xlabel('Singular vector', fontsize = 30)
    plt.ylabel('Loadings', fontsize = 30)
    
    plt.title(f'Signature for a random test sample, truth = {true_label}', fontsize = 40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()

#%%

X_train = eigendigits.T @ L
X_train = X_train.T # To make samples as rows and features as columns.
y_train = y_tr_org

X_test = eigendigits.T @ test_results
X_test = X_test.T # To make samples as rows and features as columns.
y_true = y_te_org

clf = DecisionTreeClassifier(max_depth = None,
                             min_samples_split = 2,
                             min_samples_leaf = 1)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Accuracy = ', accuracy_score(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#%%
'''
vals = proj[:,0]
colors = ['red', 'blue', 'green', 'purple', 'orange', 'magenta', 'royalblue', 'gray', 'wheat', 'greenyellow']
all_colors = []
# labels = []
for ix, ind in enumerate(y_tr_org):
    all_colors.append(colors[int(ind)])
    # labels
        

plt.figure(figsize=(30, 25))  # Optional: set the figure size
plt.bar(np.arange(len(y_tr_org)), vals, color = all_colors)#, label = y_tr_org)
plt.xlabel('Training weights')
plt.ylabel('Loadings')
plt.title(f'Components for each training digit, truth = {true_label}')
# plt.legend()
# plt.tight_layout()
plt.show()
#%%
# Find the index of a sample representing some digit.
digit_desired = 8

for index_of_interest in np.where(y_te_org == digit_desired)[0]:
# index_of_interest = np.random.choice(np.where(y_te_org == digit_desired)[0])
    comps = components[:, index_of_interest]
    
    all_colors = []
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'magenta', 'royalblue', 'gray', 'wheat', 'greenyellow']
    label_counts = dict() 
    label_counts_ls = dict() 

    vals = []
    for ix, ind in enumerate(y_tr_org):
        vals.append(comps[ix])
        all_colors.append(colors[int(ind)])
        
        if str(ind) in label_counts:     
            label_counts[str(ind)] += comps[ix]
        else:
            label_counts[str(ind)] = comps[ix]
            
        if str(ind) in label_counts_ls:     
            label_counts_ls[str(ind)].append(comps[ix])
        else:
            label_counts_ls[str(ind)] = [comps[ix]]

        # median
        label_counts_median = {}
        for k, v in label_counts_ls.items():
            label_counts_median[k] = np.median(v)
            
        # max
        label_counts_max = {}
        for k, v in label_counts_ls.items():
            label_counts_max[k] = np.max(v)
    
    print('Most similar digit by sum = ', max(label_counts, key = label_counts.get))
    print('Most similar digit by median = ', max(label_counts_median, key = label_counts_median.get))
    print('Most similar digit by max = ', max(label_counts_max, key = label_counts_max.get))

    
    
    # print('Most similar digit = ', max(label_counts, key = label_counts.get))
    
    # test_comp_df = pd.DataFrame({'similarity': list(label_counts.values()),
    #                              'digit': list(label_counts.keys()),
    #                              'color': colors})
    # ax = test_comp_df.plot.bar()
    # plt.show()

    vals = np.array(list(label_counts.values()))
    plt.figure(figsize=(30, 25))  # Optional: set the figure size
    plt.bar(labels, vals, color = colors, label = labels)
    plt.xlabel('Bar Number')
    plt.ylabel('Total Similarity')
    plt.title('Normalized Sum of components for each training digit')
    plt.legend()
    plt.tight_layout()
    plt.show()

    vals = np.array(list(label_counts_median.values()))
    plt.figure(figsize=(30, 25))  # Optional: set the figure size
    plt.bar(labels, vals, color = colors, label = labels)
    plt.xlabel('Bar Number')
    plt.ylabel('Median')
    plt.title('Median of components for each training digit')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    vals = np.array(list(label_counts_max.values()))
    plt.figure(figsize=(30, 25))  # Optional: set the figure size
    plt.bar(labels, vals, color = colors, label = labels)
    plt.xlabel('Bar Number')
    plt.ylabel('Max')
    plt.title('Max of components for each training digit')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    print('-'*50)
'''
