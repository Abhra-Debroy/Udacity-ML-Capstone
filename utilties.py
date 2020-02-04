## Disclaimer : Reuse of utility library from Term 1 course example 
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score




# my_pca function was taken from class notes and altered

def my_pca(data, n_components = None):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    INPUT: n_components - int - the number of principal components , default is None

    OUTPUT: pca - the pca object after transformation
            data_pca - the transformed data matrix with new number of components
    '''
    pca = PCA(n_components)
    data_pca = pca.fit_transform(data)
    return pca, data_pca




def plot_pca(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA i
            
    OUTPUT:
            None
    '''
    n_components=len(pca.explained_variance_ratio_)
    ind = np.arange(n_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(n_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    plt.title('Explained Variance vs Principal Component')

def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
		'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = list(good_data.keys()))
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)

def features_of_component(pca, component, column_names):
    '''
    Map weights for the component number to corresponding feature names
    
    Input:
        pca : the pca object after transformation
        component (int): component number to map to feature
        column_names (list(str)): column names of DataFrame for dataset
    
    Output:
        df_features (DataFrame): DataFrame with feature weight sorted by feature name
        
    '''
    weight_array = pca.components_[component]
    df_features = pd.DataFrame(weight_array, index=column_names, columns=['Weight'])
    variance = pca.explained_variance_ratio_[component].round(2)*100
    df_features= df_features.sort_values(by='Weight',ascending=False).round(2)
    return  variance, df_features


def get_minikmeans_score(data, nu_clusters, batch_size=50000):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = MiniBatchKMeans(n_clusters=nu_clusters, batch_size=batch_size, random_state=42)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))
    
    # both inertia_ and score gives same output
    #inertia = model.inertia_      
    return score

def get_minikmeans_silhouette_score(data, nu_clusters, batch_size=50000, sample_size =50000):
    '''
    returns the kmeans silhouette_score ts to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - silhouette_score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = MiniBatchKMeans(n_clusters=nu_clusters, batch_size=batch_size, random_state=42)
    
    
    # Then fit the model to your data using the fit method
    #kmeans.fit(data)
    #cluster_labels = kmeans.labels
    
    cluster_labels = kmeans.fit_predict(data)  

    # Obtain a score related to the model fit
       
    s_score = silhouette_score(data, cluster_labels,sample_size=sample_size, metric='euclidean',random_state=42)
    
    return s_score


    
def draw_learning_curves(estimator, x, y, training_num):
    
    '''
    Draw learning curve that shows the validation and training auc_score of an estimator 
    for training samples.
    
    Input:
        X: dataset
        y: target
        estimator: object type that implements the “fit” and “predict” methods
        training_num (int): number of training samples to plot
        
    Output:
        None
        
        '''
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, x, y, train_sizes =
    np.linspace(0.1, 1.0, training_num),cv = None, scoring = 'roc_auc')
    
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    
    plt.grid()
   
    plt.ylabel('AUROC', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for ' + str(estimator).split('(')[0] 
    plt.title(title, fontsize = 15, y = 1.03)
    
    plt.plot(np.linspace(.1, 1.0, training_num)*100, train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(np.linspace(.1, 1.0, training_num)*100, validation_scores_mean, 'o-', color="y",
             label="Cross-validation score")

    plt.yticks(np.arange(0.45, 1.02, 0.05))
    plt.xticks(np.arange(0., 100.05, 10))
    plt.legend(loc="best")
    
    print("")
    plt.show()
    print("Roc_auc train score = {}".format(train_scores_mean[-1].round(2)))
    print("Roc_auc validation score = {}".format(validation_scores_mean[-1].round(2)))
