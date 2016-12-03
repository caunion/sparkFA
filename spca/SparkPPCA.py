import logging
from PCAUtils import PCAUtils
import os.path as path
import numpy as np
from pyspark.mllib.linalg import Matrices, SparseVector, Vectors
import pyspark


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

        print ("Rows: {0}, Cols: {1}".format(nRows, nCols))

        # The two PPCA variables that improve over each iteration: Principal component
        # matrix (C) and random variance (ss), initialized randomly for the first iteration
        ss = PCAUtils.randSS()
        centralC = PCAUtils.randomMatrix(nCols, nPCs)

        # 1. Mean job: This job calculates the mean and span of the columns of the input
        # RDD<pyspark.mllib.linalg.Vector>
        matrixAccumY = sc.accumulator(np.zeros([1, nCols]), VectorAccumulatorAbsParam())
        internalSumY = np.zeros([1, nCols])

        def func1(iter):
            for yi in iter:
                indices = yi.indices()
                internalSumY[indices] += yi[indices]
            matrixAccumY.add(internalSumY)
        vectors.foreachPartition(func1)
        # end mean job



class MatrixAccumulatorParam(pyspark.accumulator.AccumulatorParam):

    def addInPlace(self, mat1, mat2):
        mat1 += mat2
        return mat1

    def zero(self, mat):
        return mat

    def addAccumulator(self, mat1, mat2):
        return self.addInPlace(mat1, mat2)

class VectorAccumulatorAbsParam(pyspark.accumulator.AccumulatorParam):
    """
        parameters should all be vector!
    """
    def addInPlace(self, vec1, vec2):
        vec1 += vec2
        return vec1
    def zero(self, vec):
        return vec
    def addAccumulator(self, vec1, vec2):
        return self.addInPlace(vec1, vec2)