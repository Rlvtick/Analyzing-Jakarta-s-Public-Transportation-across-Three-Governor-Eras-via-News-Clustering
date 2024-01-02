"""
# **A. TF-IDF + PCA**

# **Deep Embedded Clustering (DEC)**

## DEC Clustering (Jokowi)

### Autocoder and Encoder
"""

import numpy as np

# Load the preprocessed data
file_path = '/content/PCA_result_jokowi.npy'
X_Jokowi_PCA = np.load(file_path)

# Get the shape of the data
data_shape = X_Jokowi_PCA.shape

# Print the shape
print("Shape of the data:", data_shape)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/PCA_result_jokowi.npy'
X_Jokowi_PCA = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Jokowi_PCA = scaler.fit_transform(X_Jokowi_PCA)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Jokowi_PCA)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Jokowi_PCA.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Jokowi_PCA.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Jokowi_PCA, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Jokowi_PCA)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Jokowi_PCA)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Jokowi_PCA, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Jokowi_PCA)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Jokowi_PCA, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Jokowi_PCA, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Jokowi_PCA, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## DEC Clustering (Anies)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/PCA_result_anies.npy'
X_Anies_PCA = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Anies_PCA = scaler.fit_transform(X_Anies_PCA)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=30, random_state=42)
y_pred_init = kmeans.fit_predict(X_Anies_PCA)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Anies_PCA.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Anies_PCA.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Anies_PCA, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Anies_PCA)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Anies_PCA)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Anies_PCA, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Anies_PCA)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Anies_PCA, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Anies_PCA, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Anies_PCA, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## DEC Clustering (Ahok)

### Autocer and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/PCA_result_ahok.npy'
X_Ahok_PCA = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Ahok_PCA = scaler.fit_transform(X_Ahok_PCA)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=30, random_state=42)
y_pred_init = kmeans.fit_predict(X_Ahok_PCA)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Ahok_PCA.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Ahok_PCA.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Ahok_PCA, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Ahok_PCA)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Ahok_PCA)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Ahok_PCA, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Ahok_PCA)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Ahok_PCA, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Ahok_PCA, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Ahok_PCA, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""# **HDBSCAN**

## Installing Libraries
"""

#!pip install hdbscan

#!pip install scikit-learn
#!pip install sastrawi

"""## Clustering (Jokowi)"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import OneHotEncoder

# Dataset TF-DF & PCA Done
file_path = '/content/PCA_result_jokowi.npy'
X_Jokowi_PCA = np.load(file_path)
print(X_Jokowi_PCA)

num_clusters = 36

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Jokowi_PCA)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Jokowi_PCA, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Jokowi_PCA, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Jokowi_PCA, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## Clustering (Anies)"""

# Dataset TF-DF & PCA Done
file_path = '/content/PCA_result_anies.npy'
X_Anies_PCA = np.load(file_path)
print(X_Anies_PCA)

num_clusters = 30

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Anies_PCA)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Anies_PCA, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Anies_PCA, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Anies_PCA, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## Clustering (Ahok)"""

# Dataset TF-DF & PCA Done
file_path = '/content/PCA_result_ahok.npy'
X_Ahok_PCA = np.load(file_path)
print(X_Ahok_PCA)

num_clusters = 30

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Ahok_PCA)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Ahok_PCA, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Ahok_PCA, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Ahok_PCA, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""# **B. MPNet**

# Deep Embedded Clustering (DEC)

## DEC Clustering (Jokowi)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/embed_MPNet_jokowi.npy'
X_Jokowi_embed = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Jokowi_embed = scaler.fit_transform(X_Jokowi_embed)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Jokowi_embed)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Jokowi_embed.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Jokowi_embed.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Jokowi_embed, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Jokowi_embed)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Jokowi_embed)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Jokowi_embed, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Jokowi_embed)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Jokowi_embed, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Jokowi_embed, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Jokowi_embed, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## DEC Clustering (Anies)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/embed_MPNet_anies.npy'
X_Anies_embed = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Anies_embed = scaler.fit_transform(X_Anies_embed)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Anies_embed)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Anies_embed.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Anies_embed.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Anies_embed, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Anies_embed)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Anies_embed)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Anies_embed, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Anies_embed)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Anies_embed, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Anies_embed, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Anies_embed, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## DEC Clustering (Ahok)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/embed_MPNet_ahok.npy'
X_Ahok_embed = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Ahok_embed = scaler.fit_transform(X_Ahok_embed)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Anies_embed)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Ahok_embed.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Ahok_embed.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Ahok_embed, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Ahok_embed)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Ahok_embed)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Ahok_embed, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Ahok_embed)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Ahok_embed, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Ahok_embed, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Ahok_embed, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""#HBDSCAN

## Clustering (Jokowi)
"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import OneHotEncoder

# Dataset TF-DF & PCA Done
file_path = '/content/embed_MPNet_jokowi.npy'
X_Jokowi_embed = np.load(file_path)
print(X_Jokowi_embed)

num_clusters = 36

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Jokowi_embed)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Jokowi_embed, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Jokowi_embed, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Jokowi_embed, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## Clustering (Anies)

"""

# Dataset TF-DF & PCA Done
file_path = '/content/embed_MPNet_anies.npy'
X_Anies_embed = np.load(file_path)
print(X_Anies_embed)

num_clusters = 30

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Anies_embed)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Anies_embed, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Anies_embed, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Anies_embed, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## Clustering (Ahok)"""

# Dataset TF-DF & PCA Done
file_path = '/content/embed_MPNet_ahok.npy'
X_Ahok_embed = np.load(file_path)
print(X_Ahok_embed)

num_clusters = 30

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Ahok_embed)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Ahok_embed, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Ahok_embed, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Ahok_embed, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""# C. DistilBERT

## DEC Clustering (Jokowi)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/Jokowi_embed.npy'
X_Jokowi_distilbert = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Jokowi_distilbert = scaler.fit_transform(X_Jokowi_distilbert)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Jokowi_distilbert)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Jokowi_distilbert.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Jokowi_distilbert.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Jokowi_distilbert, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Jokowi_distilbert)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Jokowi_distilbert)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Jokowi_distilbert, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Jokowi_distilbert)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Jokowi_distilbert, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Jokowi_distilbert, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Jokowi_distilbert, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## DEC Clustering (Anies)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/Anies_embed.npy'
X_Anies_distilbert = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Anies_distilbert = scaler.fit_transform(X_Anies_distilbert)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Anies_distilbert)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Anies_distilbert.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Anies_distilbert.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Anies_distilbert, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Anies_distilbert)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Anies_distilbert)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Anies_distilbert, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Anies_distilbert)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Anies_distilbert, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Anies_distilbert, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Anies_distilbert, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## DEC Clustering (Ahok)

### Autocoder and Encoder
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = '/content/Ahok_embed.npy'
X_Ahok_distilbert = np.load(file_path)

# Standardize the data
scaler = StandardScaler()
X_Ahok_distilbert = scaler.fit_transform(X_Ahok_distilbert)

# K-means clustering for initialization
kmeans = KMeans(n_clusters=36, random_state=42)
y_pred_init = kmeans.fit_predict(X_Ahok_distilbert)

# Autoencoder model for feature extraction
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

input_text = Input(shape=(X_Ahok_distilbert.shape[1],))
encoded = Dense(256, activation='relu')(input_text)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(X_Ahok_distilbert.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_text, outputs=decoded)
encoder = Model(inputs=input_text, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Split the data for training and validation
X_train, X_val = train_test_split(X_Ahok_distilbert, test_size=0.1, random_state=42)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=100, batch_size=512, shuffle=True, validation_data=(X_val, X_val))

# Extract features using encoder
features = encoder.predict(X_Ahok_distilbert)

# Initialize cluster centers using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=50)
y_pred = kmeans.fit_predict(features)

"""### Custom Clustering Layer"""

# Custom Clustering Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=0.01, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape):
        input_dim = input_shape[1]
        if self.initial_weights is not None:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
            self.set_weights([self.initial_weights])
            del self.initial_weights
        else:
            self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

# Custom clustering layer in the model
clustering_layer = ClusteringLayer(num_clusters, name='clustering')(encoder.output)
clustering_model = Model(inputs=encoder.input, outputs=clustering_layer)

# Fine-tuning
clustering_model.compile(optimizer='adam', loss='kld')

# Use K-means cluster assignments for initializing the clustering layer
clustering_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Train clustering layer
y_pred_last = np.copy(y_pred_init)
for _ in range(100):  # adjust the number of iterations as needed
    q = clustering_model.predict(X_Ahok_distilbert)
    p = np.zeros_like(q)
    p[np.arange(len(q)), y_pred] = 1
    clustering_model.fit(X_Ahok_distilbert, p, epochs=1, batch_size=256)
    q_new = clustering_model.predict(X_Ahok_distilbert)
    y_pred = q_new.argmax(1)
    if np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0] < 0.001:
        break
    y_pred_last = np.copy(y_pred)

# Final cluster assignment
print("Final cluster assignments:", y_pred)

"""### Metric Evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette Score
silhouette_avg = silhouette_score(X_Ahok_distilbert, y_pred)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski-Harabasz Score
calinski_harabasz_avg = calinski_harabasz_score(X_Ahok_distilbert, y_pred)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies-Bouldin Score
davies_bouldin_avg = davies_bouldin_score(X_Ahok_distilbert, y_pred)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""#HBDSCAN

## Clustering (Jokowi)
"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import OneHotEncoder

# Dataset TF-DF & PCA Done
file_path = '/content/Jokowi_embed.npy'
X_Jokowi_distilbert = np.load(file_path)
print(X_Jokowi_distilbert)

num_clusters = 36

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Jokowi_distilbert)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Jokowi_distilbert, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Jokowi_distilbert, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Jokowi_distilbert, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## Clustering (Anies)"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import OneHotEncoder

# Dataset TF-DF & PCA Done
file_path = '/content/Anies_embed.npy'
X_Anies_distilbert = np.load(file_path)
print(X_Anies_distilbert)

num_clusters = 36

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Anies_distilbert)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Anies_distilbert, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Anies_distilbert, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Anies_distilbert, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")

"""## Clustering (Ahok)"""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import OneHotEncoder

# Dataset TF-DF & PCA Done
file_path = '/content/Ahok_embed.npy'
X_Ahok_distilbert = np.load(file_path)
print(X_Ahok_distilbert)

num_clusters = 30

# Perform HDBSCAN clustering with the specified number of clusters
clusterer = hdbscan.HDBSCAN(min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_Ahok_distilbert)

"""### Metric Evaluation"""

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_Ahok_distilbert, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

# Calinski Harabsz Score
from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_avg = calinski_harabasz_score(X_Ahok_distilbert, cluster_labels)
print(f"Calinski-Harabasz Score: {calinski_harabasz_avg}")

# Davies Bouldin Score
from sklearn.metrics import davies_bouldin_score
davies_bouldin_avg = davies_bouldin_score(X_Ahok_distilbert, cluster_labels)
print(f"Davies-Bouldin Score: {davies_bouldin_avg}")