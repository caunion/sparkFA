import pandas as pd
import numpy as np

# Specify the sample_size of the file you desire to read in
sample_size = 100
noise = np.loadtxt("noise"+str(sample_size)+".txt",  delimiter=' ')

# Specify the window_width you'd like to have
window_width = 4

# Calculate an entry in the covariance matrix using two cells
def cov_item(cell1, cell2, window_width):
    accum = 0
    for lag in range(-window_width, window_width+1):
        if lag == 0:
            series1 = pd.Series(cell1)
            series2 = pd.Series(cell2)
        elif lag < 0:
            series1 = pd.Series(cell1[:lag])
            series2 = pd.Series(cell2[-lag:])
        else:
            series1 = pd.Series(cell1[lag:])
            series2 = pd.Series(cell2[:-lag])
        accum += series1.cov(series2)
    return accum

# Calculating covariance matrix
def cov_mat(noise, window_width):
    mat = []
    for row in range(noise.shape[0]):
        row_result = []
        for col in range(noise.shape[0]):
            row_result.append(cov_item(noise[row,:], noise[col,:], window_width))
        mat.append(row_result)
    return np.array(mat)

# Saving the results
np.savetxt("cov_mat"+str(sample_size)+".txt", cov_mat(noise, window_width), delimiter=' ')
