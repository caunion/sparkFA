import pandas as pd
import numpy as np
from datetime import datetime

def one_pass(sample_number):

    df = pd.read_csv('sweep response.csv')
    df.drop('dx', 1, inplace = True)
    df.drop('Unnamed: 0', 1, inplace = True)
    df = df[:sample_number] # Get the first $sample_number cells

    # Replace string traces with np.array traces
    for i in range(df.shape[1]): # All stimuli
        column = df[str(i)]
        for cell in range(df.shape[0]):
            str_dff = column[cell].replace('[','').replace(']','').split()
            dff = []
            for item in str_dff:
                dff.append(float(item))
            df[str(i)][cell] = dff

    # Process each cell
    table = [] # cell * sweep * time
    for cell in range(df.shape[0]):
        cell_result = []
        for i in range(df.shape[1]):
            cell_result.append(df[str(i)][cell])
        mean = np.mean(np.array(cell_result), axis = 0)
        cell_result = np.subtract(cell_result, mean)
        cell_result = np.ndarray.flatten(cell_result)
        table.append(cell_result)
    table = np.transpose(np.array(table))

    np.savetxt("noise"+str(sample_number)+".txt", table, delimiter=' ')
    print table.shape

# Running the program five times on one sample_size and output the running time
for i in range(1):
    time = datetime.now()
    one_pass(6000)
    print str(datetime.now()-time)
