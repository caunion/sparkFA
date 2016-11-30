import logging
import os.path as path
import numpy as np
from pyspark.mllib.linalg import Matrices, SparseVector, Vectors

class PCAUtils:
    logger = logging.getLogger("PCAUtils")
    random = np.random
    zero = 1e-12


    @staticmethod
    def randSS():
        return PCAUtils.random.rand()

    @staticmethod
    def randValidationSS():
        return 0.9644868606768501

    @staticmethod
    def randomMatrix( rows, cols):
        return PCAUtils.random.rand(rows, cols)

    @staticmethod
    def randomValidationMatrix(row, cols):
        ret = np.array(
                [[0.730967787376657, 0.24053641567148587, 0.6374174253501083],
                [0.5504370051176339, 0.5975452777972018, 0.3332183994766498],
                [0.3851891847407185, 0.984841540199809, 0.8791825178724801],
                [0.9412491794821144, 0.27495396603548483, 0.12889715087377673],
                [0.14660165764651822, 0.023238122483889456, 0.5467397571984656]])
        return ret

    @staticmethod
    def isPass( sampleRate):
        selectionChange = PCAUtils.random.rand()
        return selectionChange > sampleRate

    @staticmethod
    def trace( mat):
        return np.trace(mat)

    @staticmethod
    def subtract( res, subtractor):
        """ Subtract a vector from an array

        :param res:
        :param subtractor:
        :return:
        """
        n = res.size
        res = res - subtractor[0:n]
        return res

    @staticmethod
    def dot( arr1, arr2):
        """dot product of vector and array

        :param arr1:
        :param arr2:
        :return:
        """
        n = arr1.size
        ret = np.sum(arr1 * arr2[0:n])
        return ret


    # TODO(daoyuan): might need replace function
    @staticmethod
    def dotVectorArray( vector, arr2):
        array2 = np.array(arr2)
        ret = sum(vector * array2)
        return ret

    # TODO(daoyuan): might need replace this function
    @staticmethod
    def denseVectorTimesMatrix( vector, matrix, xm_mahout):
        xm_mahout[:] = vector.dot(matrix)
        return xm_mahout

    # TODO(daoyuan): Might need replace this function
    @staticmethod
    def vectorTimesMatrixTranspose( vector, matrix, resArray):
        resArray[:] = vector.dot(matrix.transpose())
        return resArray

    @staticmethod
    def sparseVectorMinusVector( sparseVector, vector, resarray, nonZeroIndices):
        """Subtract a vector from a sparse vector

        :param sparseVector:
        :param vector:
        :param resarray:
        :param nonZeroIndices:  the indices of non-zero elements in the sparse vector
        :return:
        """
        for index in nonZeroIndices:
            value = sparseVector[index]
            resarray[index] = value - vector[index]
        return resarray

    @staticmethod
    def denseVectorMinusVector( vector1, vector2, resArray):
        """ Subtract two dense vectors.

        :param vector1:  pyspark.mllib.linalg.DenseVector
        :param vector2:  np vector or DenseVector
        :param resArray: result
        :return:
        """
        resArray [:] = vector1 - vector2
        return resArray

    @staticmethod
    def outerProductWithIndices( yi, ym, xi, xm, resArray, nonZeroIndices):
        """ computes the outer (tensor) product of two vectors. The result of
            applying the outer product on a pair of vectors is a matrix

        :param yi: sparse vector (of raw input data)
        :param ym: mean vector (of data)
        :param xi: dense vector
        :param xm: mean vector
        :param resArray: result matrix (a two-dimensional array)
        :param nonZeroIndices: the indices of nonzero elements in the sparse vector
        :return:
        """

        # 1. Sum(Yi' x (Xi-Xm))
        xSize = xi.size
        for i in nonZeroIndices:
            yRow = i
            yScale = yi[yRow]
            for xCol in np.arange(xSize):
                centerVal = xi[xCol] - xm[xCol]
                resArray[yRow][xCol] += centerVal * yScale

        return resArray

    @staticmethod
    def outerProductArrayInput(yi, ym, xi, xm, resArray):
        xSize = xi.size
        ySize = yi.size
        for yRow in np.arange(ySize):
            for xCol in np.arange(xSize):
                centerVal = xi[xCol] - xm[xCol]
                resArray[yRow, xCol] += centerVal * yi[yRow]
        return resArray


    @staticmethod
    def updateXtXAndYtx(realCenterYtx, realCenterSumX, ym, xm, nRows):
        """

          Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)

          M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)

          The second part is done in this function

        :param realCenterYtx:
        :param realCenterSumX:
        :param ym:
        :param xm:
        :param nRows:
        :return:
        """

        for yRow in np.arange(ym.size):
            scale =  ym[yRow]
            for xCol in np.arange(realCenterSumX.size):
                centerVal = realCenterSumX[xCol] - nRows * xm[xCol]
                currVal = realCenterYtx[yRow, xCol]
                currVal -= centerVal * scale
                realCenterYtx[yRow, xCol] = currVal
        return realCenterYtx



    @staticmethod
    def sparseVectorTimesMatrix(sparseVector, matrix, resArray):
        """ multiply a sparse vector by a matrix

        :param sparseVector:
        :param matrix:
        :param resArray:
        :return:
        """

        matrixCols = matrix.shape[1]
        for col in range(matrixCols):
            indices = sparseVector.indices # non-zero indices in sparse vector
            dotRes = 0
            for index in indices:
                value = sparseVector[index]
                dotRes += matrix[index, col] * value
            resArray[col] = dotRes
        return resArray


    @staticmethod
    def sparseVectorTimesMatrixAlloc(sparseVector, matrix):
        matrixCols = matrix.shape[1]
        tupleList = {}
        for col in range(matrixCols):
            indices = sparseVector.indices
            dotRes = 0
            for index in indices:
                value = sparseVector[index]
                dotRes += matrix[index, col] * value
            if ( abs(dotRes) > PCAUtils.zero):
                tupleList[col] = dotRes
        # alloc space for the sparse vector
        sparseRet = Vectors.sparse(matrixCols, tupleList)
        return sparseRet


    # TODO(daoyuan): This is inv for M like ss*I, we should implement another one for FA
    @staticmethod
    def inv(m):
        # assume m is square
        mat_i = np.eye(m.shape[0])
        res = np.linalg.solve(m, mat_i)
        # TODO(daoyuan): probabily we need toDenseMatrix func in java version?
        return res

    @staticmethod
    def eye(n):
        m = np.eye(n, n)
        m = Matrices.dense(n, n, m.flatten().tolist())
        return m

    @staticmethod
    def getMax(arr):
        maxv = np.max(arr)
        return maxv


    @staticmethod
    def convertMahoutToSparkMatrix(mahoutMatrix):
        """
        For compatible use

        :param mahoutMatrix:
        :return:
        """
        rows, cols = mahoutMatrix.shape

        # remember to take the transpose since denseMatrix is column major
        ret = Matrices.dense(rows, cols, mahoutMatrix.transpose().flatten().tolist())
        return ret


    @staticmethod
    def printMatrixToFile(m, format, outpoutPath):
        outputFilePath = path.join(outpoutPath, 'PCs.txt')
        if format.lower() == "dense":
            PCAUtils.printMatrixInDenseTestFormat(m, outputFilePath)
        elif format.lower() == "lil":
            PCAUtils.printMatrixInListOfListFormat(m, outputFilePath)
        elif format.lower() == "coo":
            PCAUtils.printMatrixInCoordinateFormat(m, outputFilePath)
        else:
            PCAUtils.logger.error("Unkown format. Must be one of {'dense', 'lil', 'coo'}")

    @staticmethod
    def printMatrixInDenseTestFormat(m, outputPath):
        """

        :param m: DenseMatrix
        :param outputPath:
        :return:
        """
        np.savetxt(outputPath, m.toArray())

    @staticmethod
    def printMatrixInListOfListFormat(m, outputPath):
        with open(outputPath, 'w+') as fid:
            for i in range(m.numRows):
                firstValue = True
                buffer = "{"
                for j in range(m.numCols):
                    val = m[i, j]
                    if abs(val) > PCAUtils.zero:
                        if firstValue:
                            buffer += (str(j) + ":" + str(val))
                            firstValue = False
                        else:
                            buffer += ("," + str(j) + ":" + str(val))
                buffer += "}\n"
                fid.write(buffer)

    @staticmethod
    def printMatrixInCoordinateFormat(m, ooutputPath):
        with open(ooutputPath, 'w+') as fid:
            buffer = ""
            for i in range(m.numRows):
                for j in range(m.numCols):
                    val = m[i, j]
                    if abs(val)> PCAUtils.zero:
                        buffer += ("%d,%d,%f\n".format(i, j, val))
            fid.write(buffer)
