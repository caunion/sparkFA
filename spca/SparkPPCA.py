import logging
import os.path as path
import numpy as np
from pyspark.mllib.linalg import Matrices, SparseVector, Vectors

class SparkPPCA:
    logger = logging.getLogger("SparkPPCA")
    MAX_ROUND = 100
    CALCULATE_ERR_ATTHEEND = False

    @staticmethod
    def computePrincipalComponents(sc, inputMatrix, outputPath, nRows, nCols, nPCs, errRate, maxIteration, computeProjectedMatrix):
        # TODO(daoyuan): convert inputMatrix to RDD format
        pass


    @staticmethod
    def computePCA(sc, vectors, outputPath, nRows, nCols, nPCs, errRate, maxIteration, computeProjectedMatrix):
        """core func

        :param sc: spart context
        :param vectors:  RDD of vectors representing the rows of input matrix
        :param outputPath: path to save
        :param nRows:  Number of rows in input matrix
        :param nCols:  Number of cols in input matrix
        :param nPCs:  Number of desireed principal components
        :param errRate:  The sampling rate that is used for computing the reconstruction error
        :param maxIteration:  Maximum number of iteration before terminating
        :param computeProjectedMatrix:
        :return:
        """
        pass