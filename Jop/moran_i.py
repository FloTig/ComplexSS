from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import scipy.sparse as sps

def get_initial_indices(flat_index_1, flat_index_2, n, m):
    """
    Convert two flattened array indices back to its original (i, j) tuples. Both indices should come from the same flattened array, or two arrays of the same length.
    
    Parameters:
        flat_index_1 : int
        flat_index_2 : int
        n : int, length of original array
        m : int, width of original array
    
    Returns:
        tuple of two original coordinate tuples of ints
    """
    i1 = flat_index_1 // n
    j1 = flat_index_1 % m
    i2 = flat_index_2 // n 
    j2 = flat_index_2 % m
    return ((i1, j1), (i2, j2))

def compute_weights_matrix(shape, radius, save_matrix=False, file_name=None):
    """
    Computes a weights matrix with the inverse distance rule.

    Parameters:
        shape : tuple of ints, shape of the array to compute weights matrix of.
        radius : square radius of neighbors to consider. 

    Returns:
        weight_matrix : sparse matrix in coo format.
    """
    if save_matrix and type(file_name) != str:
        raise ValueError(f"file name must be of type string.")
    n, m = shape
    N = n * m
    nonzero_weights = []
    
    for i in range(n):
        for j in range(m):
            neighbor_range = get_coordinate_range((i, j), radius, shape)
            nonzero_weights.extend(get_weight(shape, (i, j), neighbor_range))
    nonzero_i, nonzero_j, nonzero_data = zip(*nonzero_weights)
    weight_matrix = sps.coo_array((nonzero_data, (nonzero_i, nonzero_j)), dtype=float)

    if save_matrix:
        sps.save_npz(file_name, weight_matrix)

    return weight_matrix

def get_coordinate_range(center, radius, shape):
    """
    Compute range of indices that signify neighbors that get nonzero weights.
    
    Parameters:
        center : tuple of ints, current coordinate.
        radius : square radius of neighbors to allow a non zero weight.
        shape : tuple of ints, shape of the array, needed to prevent out of bounds indices

    Returns:
        min_i : int, minimum row index that can be considered a neighbor.  
        max_i : int, maximum row index that can be considered a neighbor.
        min_j : int, minimum column index that can be considered a neighbor.
        max_j : int, maximum column index that can be considered a neighbor.
    """
    n, m = shape
    i, j = center
    min_i =  max(0, i - radius)
    max_i = min(n, i + radius)
    min_j =  max(0, j - radius)
    max_j = min(m, j + radius)

    return min_i, max_i, min_j, max_j

def get_weight(shape, center, neighbor_range):
    """
    Compute neighbouring weights for a range of neighbors from a specific coordinate

    Parameters:
        shape : tuple of ints, shape of the array, needed to prevent out of bounds indices
        center : tuple of ints, current coordinate.
        neighbor_range : tuple of ints representing the minimum and maximum row and column indices to consider neighbors
    """
    min_i, max_i, min_j, max_j = neighbor_range
    n, m = shape
    current_i, current_j = center
    flat_i = current_i * m + current_j
    nonzero_weights = []
    for i in range(min_i, max_i):
        for j in range(min_j, max_j):
            flat_j = i * m + j
            di = i - current_i
            dj = j - current_j
            if not (di + dj == 0):
                w = 1 / np.sqrt((di**2 + dj**2))
                sparse_tuple = (flat_i, flat_j, w)     
                nonzero_weights.append(sparse_tuple)     
    return nonzero_weights

def moran_denominator(flat_array):
    """Compute the Moran denominator for an array from a flattend array. Returns the Moran denominator"""
    sq_dist = 0
    x_mean = np.mean(flat_array)
    for xi in flat_array:
        sq_dist += (xi-x_mean)**2
    return sq_dist

def compute_moran_i(array, i, j, value, n, m, x_mean, denominator):
    """
    Compute Moran I value for a single flattened array indices pair.
    
    Parameters:
        array : array to perform Moran's I metric on.
        i : int, first index of flattened array
        j : int, second index of flattend array
        value : float, weight value in weight matrix
        n : int, length of original array
        m : int, width of original array
        x_mean : float, mean value of origin array
        denominator : float, value of the Moran's I denominator
    
        Returns: 
            float, unnormalized Moran's I value of one grid cell with another grid cell

    """
    (i1, j1), (i2, j2) =  get_initial_indices(i, j, n, m)
    return value * (array[i1,j1]-x_mean)*(array[i2,j2]-x_mean) / denominator


def parallel_compute_moran_i(array, sparse_matrix_chunk, n, m, x_mean, denominator):
    """
    Compute Moran I value for a set of flattened array index pairs.

    Parameters:
        array : array to perform Moran's I metric on.
        sparse_matric_chunk : list of coordinated to compute Moran's I value for.
        n : int, length of original array
        m : int, width of original array
        x_mean : float, mean value of origin array
        denominator : float, value of the Moran's I denominator

    Returns:
        float : unnormalized sum of Moran's I values for selected coordinates
    """
    value = 0
    for i, j, weight in sparse_matrix_chunk:
        (i1, j1), (i2, j2) =  get_initial_indices(i, j, n, m)
        value += weight * (array[i1,j1]-x_mean)*(array[i2,j2]-x_mean) / denominator
    return value

def moran_i(array, weights=None, r=10):
    """
    Function tht computes the Moran's I value of an array.

    Parameters:
        array : array to perform Moran's I metric on.
        
        weights : sparse matrix in coo format, weights matrix. Defaults to None, in which case the matrix is computed.
        r : int, square range of neighbors in case the weight matrix needs to be computed.
    
    Returns:
        float, normalized Moran's I value.
    """

    n, m = array.shape
    N = array.size

    try:
        weights = sps.load_npz(weights)
    except (TypeError, FileNotFoundError):
        print(f"Weights file does not exist, constructing weight matrix {array.shape} shaped array")
        weights = compute_weights_matrix((n, m), r)

    x_mean = np.mean(array)
    I = 0
    flat_array = array.flatten()
    denominator = moran_denominator(flat_array)
    
    N = array.size
    n = array.shape[0]
    m = array.shape[1]
    x_mean = np.mean(array)
    I = 0
    
    flat_array = array.flatten()
    denominator = moran_denominator(flat_array)
    for i, j, value in zip(weights.row, weights.col, weights.data):
        I += compute_moran_i(array, i, j, value, n, m, x_mean, denominator)
    return N/np.sum(weights.data) * I


def split_coo_matrix(coo_matrix, n_jobs):
    """
    Function that splits the coordinates present in the sparse weight matrix in (almost) even chunks in case of parallel computation of Moran's I value.

    Parameters:
        coo_matrix : sparse weight matrix in coo format.
        n_jobs : int, amount of jobs to be scheduled
    
        Returns:
            chunks : list of coordinate tuples, representing the coordinates to be used by a job
    """
    coo_matrix_list = list(zip(coo_matrix.row, coo_matrix.col, coo_matrix.data))
    chunk_size = len(coo_matrix_list) // (n_jobs + 0)

    chunks = []
    for i in range(n_jobs-1):  
        chunks.append(coo_matrix_list[i*chunk_size:(i+1)*chunk_size])
    chunks.append(coo_matrix_list[(n_jobs-1)*chunk_size::])
       
    return chunks


def parallel_moran_i(array, n_jobs, weights=None, r=10):
    """
    Function tht computes the Moran's I value of an array in a parallel fashion.

    Parameters:
        array : array to perform Moran's I metric on.
        n_jobs : int, amount of jobs to be schedule.
    
    Keyword arguments:        
        weights : sparse matrix in coo format, weights matrix. Defaults to None, in which case the matrix is computed.
        r : int, square range of neighbors in case the weight matrix needs to be computed.
    
    Returns:
        float, normalized Moran's I value.
    """
    n, m = array.shape
    N = array.size
    try:
        weights = sps.load_npz(weights)
    except (TypeError, FileNotFoundError):
        print(f"Weights file does not exist, constructing weight matrix {array.shape} shaped array")
        weights = compute_weights_matrix((n, m), r)

    x_mean = np.mean(array)
    I = 0
    flat_array = array.flatten()
    denominator = moran_denominator(flat_array)
    split_weights = split_coo_matrix(weights, n_jobs*4)

    results = Parallel(n_jobs=n_jobs, require='sharedmem', verbose=1)(
        delayed(parallel_compute_moran_i)(array, chunk, n, m, x_mean, denominator)
        for chunk in split_weights)
    
    I = sum(results)
    return N / np.sum(weights.data) * I

