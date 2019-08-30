import numpy as np
from sklearn.decomposition import PCA


''' This is an example code for PCA projection
'''
def pca_basic(t2d_data_sequence):
    # pca_input: 2184*(240*320)
    
    t2d_data_height = t2d_data_sequence.shape[0]
    t2d_data_width = t2d_data_sequence.shape[1]
    t2d_data_sequence = t2d_data_sequence.reshape((t2d_data_height*t2d_data_width, np.size(t2d_data_sequence, 2)))
    t2d_data_sequence = np.transpose(t2d_data_sequence)

    # Eigenfaces
    output_pca = PCA(n_components=0.95)
    
    # PCA projection
    _ = output_pca.fit(t2d_data_sequence)
    t2d_data_sequence = output_pca.transform(t2d_data_sequence)

    # Eigenfaces
    eigen_faces = output_pca.components_
    var_percent = output_pca.explained_variance_ratio_
    
    return eigen_faces, var_percent