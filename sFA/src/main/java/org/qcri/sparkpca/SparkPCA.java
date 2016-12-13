/**
 * QCRI, sPCA LICENSE
 * sPCA is a scalable implementation of Principal Component Analysis (PCA) on of Spark and MapReduce
 *
 * Copyright (c) 2015, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 */
package org.qcri.sparkpca;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;

import com.sun.javafx.stage.FocusUngrabEvent;
import org.apache.commons.cli.*;
import org.apache.hadoop.hdfs.server.common.Storage.StorageState;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.ui.storage.StoragePage;
import org.apache.spark.util.SystemClock;
import org.qcri.sparkpca.FileFormat.OutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.instrument.Instrumentation;

import scala.Tuple2;

/**
 * This code provides an implementation of PPCA: Probabilistic Principal
 * Component Analysis based on the paper from Tipping and Bishop:
 *
 * sPCA: PPCA on top of Spark
 *
 *
 * @author Tarek Elgamal
 *
 */

public class SparkPCA implements Serializable {

	private final static Logger log = LoggerFactory.getLogger(SparkPCA.class);
	final int MAX_ROUNDS=100;
	private final static boolean CALCULATE_ERR_ATTHEEND = false;

	public static void main(String[] args) {


		//Parsing input arguments
		final String inputPath;
		final String outputPath;
		final int nRows;
		final int nCols;
		final int nPCs;
		CommandLine commandLine;
		CommandLineParser parser = new GnuParser();
		Options options = new Options();
		options.addOption(OptionBuilder.withArgName("i").hasArg().withDescription("input path").create("i"));
		options.addOption(OptionBuilder.withArgName("o").hasArg().withDescription("output path").create("o"));
		options.addOption(OptionBuilder.withArgName("rows").hasArg().withDescription("rows of input").create("rows"));
		options.addOption(OptionBuilder.withArgName("cols").hasArg().withDescription("Cols/dim of the input").create("cols"));
		options.addOption(OptionBuilder.withArgName("pcs").hasArg().withDescription("pcs ").create("pcs"));
		// option parameters
		options.addOption(OptionBuilder.withArgName("errSampleRate").hasOptionalArg().withDescription("errSampleRate").create("errSampleRate"));
		options.addOption(OptionBuilder.withArgName("maxIter").hasOptionalArg().withDescription("maxIter ").create("maxIter"));
		options.addOption(OptionBuilder.withArgName("outFmt").hasOptionalArg().withDescription("outFmt ").create("outFmt"));
		options.addOption(OptionBuilder.withArgName("computeProjectedMatrix").hasOptionalArg().withDescription("computeProjectedMatrix ").create("computeProjectedMatrix"));

		try{
			commandLine = parser.parse(options, args);
		} catch (ParseException e){
			printLogMessage("commandline parser exception" + e.getMessage());
			return;
		}

		try {
			inputPath=commandLine.getOptionValue("i");
			if(inputPath==null)
				throw new IllegalArgumentException();
		}
		catch(Exception e) {
			printLogMessage("i");
			return;
		}
		try {
			outputPath=commandLine.getOptionValue("o");
			if(outputPath==null)
				throw new IllegalArgumentException();
		}
		catch(Exception e) {
			printLogMessage("o");
			return;
		}

		try {
			nRows=Integer.parseInt(commandLine.getOptionValue("rows"));
		}
		catch(Exception e) {
			printLogMessage("rows");
			return;
		}

		try {
			nCols=Integer.parseInt(commandLine.getOptionValue("cols"));
		}
		catch(Exception e) {
			printLogMessage("cols");
			return;
		}
		try {
			nPCs=Integer.parseInt(commandLine.getOptionValue("pcs"));
		}
		catch(Exception e) {
			printLogMessage("pcs");
			return;
		}
		/**
		 * Defaults for optional arguments
		 */
		double errRate=1;
		int maxIterations=3;
		OutputFormat outputFileFormat=OutputFormat.DENSE;
		int computeProjectedMatrix=0;
		try {
			errRate= Float.parseFloat(commandLine.getOptionValue("errSampleRate"));
		}
		catch(Exception e) {

			int length = String.valueOf(nRows).length();
			if(length <= 4)
				errRate= 1;
			else
				errRate=1/Math.pow(10, length-4);
			log.warn("error sampling rate set to:  errSampleRate=" + errRate);
		}
		try {
			maxIterations=Integer.parseInt(commandLine.getOptionValue("maxIter"));
		}
		catch(Exception e) {
			log.warn("maximum iterations is set to default: maximum Iterations=" + maxIterations);
		}

		try {
			outputFileFormat=OutputFormat.valueOf(commandLine.getOptionValue("outFmt"));
		}
		catch(IllegalArgumentException e) {
			log.warn("Invalid Format " + System.getProperty("outFmt") +  ", Default output format"  + outputFileFormat + " will be used ");
		}
		catch(Exception e) {
			log.warn("Default output format "  + outputFileFormat + " will be used ");
		}

		try {
			computeProjectedMatrix=Integer.parseInt(commandLine.getOptionValue("computeProjectedMatrix"));
		}
		catch(Exception e) {
			log.warn("Projected Matrix will not be computed, the output path will contain the principal components only");
		}



		//Setting Spark configuration parameters
		SparkConf conf = new SparkConf().setAppName("SparkPCA");
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		JavaSparkContext sc = new JavaSparkContext(conf);

		//compute principal components
		org.apache.spark.mllib.linalg.Matrix principalComponentsMatrix = computePrincipalComponents(sc, inputPath, outputPath, nRows, nCols, nPCs, errRate, maxIterations, computeProjectedMatrix);

		//save principal components
		PCAUtils.printMatrixToFile(principalComponentsMatrix, outputFileFormat, outputPath);


		log.info("Principal components computed successfully ");
	}

	/**
	 * Compute principal component analysis where the input is a RowMatrix
	 * @param sc
	 * 	 		Spark context that contains the configuration parameters and represents connection to the cluster
	 * 			(used to create RDDs, accumulators and broadcast variables on that cluster)
	 * @param inputMatrix
	 * 			Input Matrix
	 * @param nRows
	 * 			Number of rows in input Matrix
	 * @param nCols
	 * 			Number of columns in input Matrix
	 * @param nPCs
	 * 			Number of desired principal components
	 * @param errRate
	 * 			The sampling rate that is used for computing the reconstruction error
	 * @param maxIterations
	 * 			Maximum number of iterations before terminating
	 * @return Matrix of size nCols X nPCs having the desired principal components
	 */
	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, RowMatrix inputMatrix, String outputPath, final int nRows, final int nCols, final int nPCs, final double errRate, final int maxIterations, final int computeProjectedMatrix) {
		JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors = inputMatrix.rows().toJavaRDD().cache();
		return computePrincipalComponents(sc, vectors, outputPath, nRows, nCols, nPCs, errRate, maxIterations, computeProjectedMatrix);
	}

	/**
	 * Compute principal component analysis where the input is a path for a hadoop sequence File <IntWritable key, VectorWritable value>
	 * @param sc
	 * 	 		Spark context that contains the configuration parameters and represents connection to the cluster
	 * 			(used to create RDDs, accumulators and broadcast variables on that cluster)
	 * @param inputPath
	 * 			Path to the sequence file that represents the input matrix
	 * @param nRows
	 * 			Number of rows in input Matrix
	 * @param nCols
	 * 			Number of columns in input Matrix
	 * @param nPCs
	 * 			Number of desired principal components
	 * @param errRate
	 * 			The sampling rate that is used for computing the reconstruction error
	 * @param maxIterations
	 * 			Maximum number of iterations before terminating
	 * @return Matrix of size nCols X nPCs having the desired principal components
	 */
	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, String inputPath, String outputPath, final int nRows, final int nCols, final int nPCs, final double errRate, final int maxIterations, final int computeProjectedMatrix) {


		//Read from sequence file
		JavaPairRDD<IntWritable,VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class, VectorWritable.class);


		//Convert sequence file to RDD<org.apache.spark.mllib.linalg.Vector> of Vectors
		JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors=seqVectors.map(new Function<Tuple2<IntWritable,VectorWritable>, org.apache.spark.mllib.linalg.Vector>() {

			public org.apache.spark.mllib.linalg.Vector call(Tuple2<IntWritable, VectorWritable> arg0)
					throws Exception {

				org.apache.mahout.math.Vector mahoutVector = arg0._2.get();
				Iterator<Element> elements =  mahoutVector.nonZeroes().iterator();
				ArrayList<Tuple2<Integer, Double>> tupleList = new  ArrayList<Tuple2<Integer, Double>>();
				while(elements.hasNext())
				{
					Element e = elements.next();
					if(e.index() >= nCols || e.get() == 0 )
						continue;
					Tuple2<Integer,Double> tuple = new Tuple2<Integer,Double>(e.index(), e.get());
					tupleList.add(tuple);
				}
				org.apache.spark.mllib.linalg.Vector sparkVector = Vectors.sparse(nCols,tupleList);
				return sparkVector;
			}
		}).persist(StorageLevel.MEMORY_ONLY_SER());

		return computePrincipalComponents(sc, vectors, outputPath, nRows, nCols, nPCs, errRate, maxIterations, computeProjectedMatrix);
	}
	/**
	 * Compute principal component analysis where the input is an RDD<org.apache.spark.mllib.linalg.Vector> of vectors such that each vector represents a row in the matrix
	 * @param sc
	 * 	 		Spark context that contains the configuration parameters and represents connection to the cluster
	 * 			(used to create RDDs, accumulators and broadcast variables on that cluster)
	 * @param vectors
	 * 			An RDD of vectors representing the rows of the input matrix
	 * @param nRows
	 * 			Number of rows in input Matrix
	 * @param nCols
	 * 			Number of columns in input Matrix
	 * @param nPCs
	 * 			Number of desired principal components
	 * @param errRate
	 * 			The sampling rate that is used for computing the reconstruction error
	 * @param maxIterations
	 * 			Maximum number of iterations before terminating
	 * @return Matrix of size nCols X nPCs having the desired principal components
	 */
	public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors, String outputPath, final int nRows, final int nCols, final int nPCs, final double errRate, final int maxIterations, final int computeProjectedMatrix) {

		System.out.println("Rows: " + nRows + ", Columns " + nCols);
		double ss;
		Matrix psi;
		Matrix centralC;
		Matrix centralCFA;
		double LL;
		double LL2;
		double LLChange = 1e6;
		double LLOld = -1e6;

		// The two PPCA variables that improve over each iteration: Principal component matrix (C) and random variance (ss).
		// Initialized randomly for the first iteration
		ss= PCAUtils.randSS();
		psi = PCAUtils.randPsi(nCols);
		//double[][] CFAtest={ {0.13,0.74,0.21}, {0.45,0.96,0.53}, {0.467,0.89,0.34}, {0.456,0.49,0.325}, {0.708,0.635,0.221}};
		//double[] psiTest={0.923,0.79,0.48,0.587,0.15};
		// psi = new DiagonalMatrix(psiTest);
		centralC= PCAUtils.randomMatrix(nCols, nPCs);
		centralCFA = centralC.clone();
		//Matrix centralCFA = new DenseMatrix(CFAtest);






		// 1. Mean Job : This job calculates the mean and span of the columns of the input RDD<org.apache.spark.mllib.linalg.Vector>
		final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[nCols], new VectorAccumulatorParam());
		final double[] internalSumY=new double[nCols];
		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0)
					throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				int[] indices=null;
				int i;
				while(arg0.hasNext())
				{
					yi=arg0.next();
					indices=((SparseVector)yi).indices();
					for(i=0; i< indices.length; i++ )
					{
						internalSumY[indices[i]]+=yi.apply(indices[i]);
					}
				}
				matrixAccumY.add(internalSumY);
			}

		});//End Mean Job

		//Get the sum of column Vector from the accumulator and divide each element by the number of rows to get the mean
		final Vector meanVector=new DenseVector(matrixAccumY.value()).divide(nRows);
		final Broadcast<Vector> br_ym_mahout = sc.broadcast(meanVector);


		//CovYDiagJob -- Compute the covariance of the input data in Spark Job
		final Accumulator<double[][]> matrixAccumYCov = sc.accumulator(new double[nCols][nCols], new MatrixAccumulatorParam());
		final Accumulator<double[][]> matrixAccumYDotYm = sc.accumulator(new double[nCols][nCols], new MatrixAccumulatorParam());


		final double[][] resArrayYCov= new double[nCols][nCols];
		final double[][] resArrayYDotYm= new double[nCols][nCols];

		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0)
					throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				int[] indices=null;
				while(arg0.hasNext())
				{
					yi=arg0.next();
					indices=((SparseVector)yi).indices();

					PCAUtils.outerProductYi(yi,resArrayYCov,indices);


					PCAUtils.outerProductYiYMean(yi,br_ym_mahout.value(),resArrayYDotYm,indices);

				}
				matrixAccumYCov.add(resArrayYCov);
				matrixAccumYDotYm.add(resArrayYDotYm);
			}

		});//YCov Job


		Matrix YCov = new DenseMatrix(matrixAccumYCov.value());
		Matrix YDotYm = new DenseMatrix(matrixAccumYDotYm.value());

		YCov = YCov.minus(YDotYm);
		Matrix YCovScale = YCov.times(1.0/(float) nRows);






/*		//2. Frobenious Norm Job : Obtain Frobenius norm of the input RDD<org.apache.spark.mllib.linalg.Vector>
		*//**
		 * To compute the norm2 of a sparse matrix, iterate over sparse items and sum
		 * square of the difference. After processing each row, add the sum of the
		 * meanSquare of the zero-elements that were ignored in the sparse iteration.
		 *//*
		double meanSquareSumTmp=0;
		for(int i=0; i < br_ym_mahout.value().size(); i++)
		{
			double element=br_ym_mahout.value().getQuick(i);
			meanSquareSumTmp+=element*element;
		}
		final double meanSquareSum= meanSquareSumTmp;
		final Accumulator<Double> doubleAccumNorm2 = sc.accumulator(0.0);
		vectors.foreach( new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {

			public void call(org.apache.spark.mllib.linalg.Vector yi)
					throws Exception {
				double norm2=0;
				double meanSquareSumOfZeroElements = meanSquareSum;
				int[] indices =  ((SparseVector)yi).indices();
				int i;
				int index;
				double v;
				//iterate over non
				for(i=0; i < indices.length ; i++) {
					index = indices[i];
					v = yi.apply(index);
					double mean = br_ym_mahout.value().getQuick(index);
					double diff = v - mean;
					diff *= diff;

					// cancel the effect of the non-zero element in meanSquareSum
					meanSquareSumOfZeroElements -= mean * mean;
					norm2 += diff;
				}
				// For all all zero items, the following has the sum of mean square
				norm2+=meanSquareSumOfZeroElements;
				doubleAccumNorm2.add(norm2);
			}

		});//end Frobenious Norm Job //I don't actually think I need this for anything...

		double norm2=doubleAccumNorm2.value();
		log.info("NOOOOORM2=" + norm2 );

		// initial CtC
		Matrix centralCtC= centralC.transpose().times(centralC);*/




		// -------------------------- EM Iterations

		Matrix centralY2X=null;
		Matrix centralY2Xm = null;
		// while count
		int round = 0;
		double prevObjective = Double.MAX_VALUE;
		double error = 0;
		double error_FA = 0;
		double relChangeInObjective = Double.MAX_VALUE;
		double prevError=Double.MAX_VALUE;
		double prevError_FA=Double.MAX_VALUE;
		final float threshold = 0.00001f;
		final int firstRound=round;

		//for (; (round < maxIterations && relChangeInObjective > threshold && prevError > 0.02); round++) {
		//for (; (round < 20 && LLChange < 1 && Double.isNaN(LLChange) != true); round++) {
			//right now this is just doing 20 its of the EM algorithm. May want to add condition to break
		for (; (round < maxIterations && LLChange > 1); round++) { //right now this is just doing 20 its of the EM algorithm. May want to add condition to break



			// OLD sPCA code...Sx = inv( ss * eye(d) + CtC );
			//Matrix centralSx = centralCtC.clone();

			//inv(psi)
			Matrix psiInv = psi.clone();
			psiInv.viewDiagonal().assign(psi.viewDiagonal().assign(Functions.INV)); //could just do division here. Don't know how it handles elementwise inv...
			//inv(psi)*C
			Matrix psiInvC = psiInv.times(centralCFA);

			// inv(M) = ...
			Matrix V = centralCFA.transpose().times(psiInvC);//might be able to do this stuff more efficiently with Chol decomp
			V.viewDiagonal().assign(Functions.plus(1));
			V = PCAUtils.inv(V);
			Matrix mInv = psiInv.minus((psiInvC.times(V)).times(psiInvC.transpose()));

			// ... OLD sPCA code...
			//Sx = inv( eye(d) + CtC/ss );
			//centralSx.viewDiagonal().assign(Functions.plus(ss));
			//centralSx = PCAUtils.inv(centralSx);

			// ... OLD sPCA code...
			// X = Y * C * Sx' => Y2X = C * Sx'
			//centralY2X =centralC.times(centralSx.transpose());



			//Xm = C' * mInv * Y -> Y2Xm = C' * mInv
			centralY2Xm = centralCFA.transpose().times(mInv);

			//Broadcast Y2X because it will be used in several jobs and several iterations.
			//final Broadcast<Matrix> br_centralY2X = sc.broadcast(centralY2X); //old sPCA code
			final Broadcast<Matrix> br_centralY2Xm = sc.broadcast(centralY2Xm.transpose());

			// Xm = Ym * Y2X
			//Vector xm_mahout = new DenseVector(nPCs); //old sPCA code
			Vector xmFA_mahout = new DenseVector(nPCs);

			//xm_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(), centralY2X, xm_mahout); //old sPCA code

			xmFA_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(),centralY2Xm.transpose(),xmFA_mahout);

			//Broadcast Xm because it will be used in several iterations.
			//final Broadcast<Vector> br_xm_mahout = sc.broadcast(xm_mahout);//old sPCA code

			final Broadcast <Vector> br_xmFA_mahout = sc.broadcast(xmFA_mahout);

			// We skip computing X as we generate it on demand using Y and Y2X

			// 3. X'X and Y'X Job:  The job computes the two matrices X'X and Y'X
			/**
			 * Xc = Yc * MEM	(MEM is the in-memory broadcasted matrix Y2X)
			 *
			 * XtX = Xc' * Xc
			 *
			 * YtX = Yc' * Xc
			 *
			 * It also considers that Y is sparse and receives the mean vectors Ym and Xm
			 * separately.
			 *
			 * Yc = Y - Ym
			 *
			 * Xc = X - Xm
			 *
			 * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
			 *
			 * XtX = (X - Xm)' * (X - Xm)
			 *
			 * YtX = (Y - Ym)' * (X - Xm)
			 *
			 */
			//final Accumulator<double[][]> matrixAccumXtx = sc.accumulator(new double[nPCs][nPCs], new MatrixAccumulatorParam()); //old sPCA code
			//final Accumulator<double[][]> matrixAccumYtx = sc.accumulator(new double[nCols][nPCs], new MatrixAccumulatorParam()); //old sPCA code
			//final Accumulator<double[]> matrixAccumX = sc.accumulator(new double[nPCs], new VectorAccumulatorParam()); //old sPCA code

			final Accumulator <double [][]> matrixAccumSigmaXm = sc.accumulator(new double[nPCs][nPCs], new MatrixAccumulatorParam());
			final Accumulator<double[][]> matrixAccumYProj = sc.accumulator(new double[nCols][nPCs], new MatrixAccumulatorParam());
			final Accumulator<double []> matrixAccumX_FA = sc.accumulator(new double [nPCs], new VectorAccumulatorParam());





		    /*
		     * Initialize the output matrices and vectors once in order to avoid generating massive intermediate data in the workers
		     */
			//old sPCA code (next 3 lines)
			//final double[][] resArrayYtX = new double[nCols][nPCs];
			//final double[][] resArrayXtX = new double[nPCs][nPCs];
			//final double[]   resArrayX = new double[nPCs];

			final double[][] resArrayYproj = new double[nCols][nPCs];
			final double[][] resArraySigmaXm = new double[nPCs][nPCs];
			final double[]   resArrayX_FA = new double[nPCs];

		    /*
		     * Used to sum the vectors in one partition.
		    */
			//old sPCA code (next 3 lines)
			//final double[][] internalSumYtX = new double[nCols][nPCs];
			//final double[][]   internalSumXtX = new double[nPCs][nPCs];
			//final double[]   internalSumX = new double[nPCs];

			final double[][] internalSumYProj= new double[nCols][nPCs];
			final double[][] internalSumSigmaXm = new double[nPCs][nPCs];
			final double[]   internalSumX_FA = new double[nPCs];

			vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

				public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
					org.apache.spark.mllib.linalg.Vector yi;
					while(arg0.hasNext())
					{
						yi=arg0.next();

		    			/*
		    		     * Perform in-memory matrix multiplication xi = yi' * Y2X
		    		    */
						//PCAUtils.sparseVectorTimesMatrix(yi,br_centralY2X.value(), resArrayX); //old sPCA code

						PCAUtils.sparseVectorTimesMatrix(yi,br_centralY2Xm.value(), resArrayX_FA);

						//get only the sparse indices
						int[] indices=((SparseVector)yi).indices();

						//old sPCA code (next 2 lines)
						//PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayX, br_xm_mahout.value(), resArrayYtX,indices);
						//PCAUtils.outerProductArrayInput(resArrayX, br_xm_mahout.value(), resArrayX, br_xm_mahout.value(), resArrayXtX);

						PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayX_FA, br_xmFA_mahout.value(), resArrayYproj, indices);
						PCAUtils.outerProductArrayInput(resArrayX_FA, br_xmFA_mahout.value(), resArrayX_FA, br_xmFA_mahout.value(), resArraySigmaXm);

						int i,j,rowIndexYtX;

						//add the sparse indices only
						for(i=0; i< indices.length; i++ )
						{
							rowIndexYtX=indices[i];
							for(j=0; j< nPCs; j++ )
							{
								//old sPCA code (next 2 lines)
								//internalSumYtX[rowIndexYtX][j]+=resArrayYtX[rowIndexYtX][j];
								//resArrayYtX[rowIndexYtX][j]=0; //reset it

								internalSumYProj[rowIndexYtX][j]+=resArrayYproj[rowIndexYtX][j];
								resArrayYproj[rowIndexYtX][j] = 0;
							}

						}
						for(i=0; i< nPCs; i++ )
						{
							//internalSumX[i]+=resArrayX[i];
							for(j=0; j< nPCs; j++ )
							{
								//old sPCA code (next 2 lines)
								//internalSumXtX[i][j]+=resArrayXtX[i][j];
								//resArrayXtX[i][j]=0; //reset it

								internalSumSigmaXm[i][j]+=resArraySigmaXm[i][j];
								resArraySigmaXm[i][j]=0;
							}

						}
					}
					//old sPCA code (next 3 lines)
					//matrixAccumX.add(internalSumX);
					//matrixAccumXtx.add(internalSumXtX);
					//matrixAccumYtx.add(internalSumYtX);

					matrixAccumX_FA.add(internalSumX_FA);
					matrixAccumSigmaXm.add(internalSumSigmaXm);
					matrixAccumYProj.add(internalSumYProj);
				}

			});//end  X'X and Y'X Job


		   /*
		    * Get the values of the accumulators.
		   */
			//old sPCA code (next 3 lines)
			//Matrix centralYtX = new DenseMatrix(matrixAccumYtx.value());
			//Matrix centralXtX = new DenseMatrix(matrixAccumXtx.value());
			//Vector centralSumX= new DenseVector(matrixAccumX.value());

			Matrix centralYProj = new DenseMatrix(matrixAccumYProj.value());
			Matrix centralSigmaXm = new DenseMatrix(matrixAccumSigmaXm.value());
			Vector centralSumX_FA = new DenseVector(matrixAccumX_FA.value());


		   /*
	        * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
	        *
	        * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
	        *
	        * The first part is done in the previous job and the second in the following method
	       */
			//old sPCA code and I don't even know what this does...I don't replicate it.
			//centralYtX= PCAUtils.updateXtXAndYtx(centralYtX, centralSumX, br_ym_mahout.value(), xm_mahout, nRows);
			//centralXtX= PCAUtils.updateXtXAndYtx(centralXtX, centralSumX, xm_mahout, xm_mahout, nRows);

			//need to scale outerproduct results by number of elements
			centralYProj = centralYProj.times(1.0/(float) nRows);
			centralSigmaXm = centralSigmaXm.times(1.0/(float) nRows);

/*			// OLD sPCA code....XtX = X'*X + ss * Sx
			final double finalss = ss;
			centralXtX.assign(centralSx, new DoubleDoubleFunction() {
				@Override
				public double apply(double arg1, double arg2) {
					return arg1 + finalss * arg2;
				}
			});*/


			centralSigmaXm.viewDiagonal().assign(Functions.plus(1));
			centralSigmaXm = centralSigmaXm.minus(centralY2Xm.times(centralCFA));


			// OLD spCA Code, next 3 lines...C = (Ye'*X) / SumXtX;
			//Matrix invXtX_central = PCAUtils.inv(centralXtX);
			//centralC = centralYtX.times(invXtX_central);
			//centralCtC = centralC.transpose().times(centralC);


			Matrix invSigmaXm = PCAUtils.inv(centralSigmaXm); //might be a way to make this stuff more efficient...
			centralCFA = centralYProj.times(invSigmaXm);

			//all this code is for old sPCA stuff
			// Compute new value for ss
			// ss = ( sum(sum(Ye.^2)) + trace(XtX*CtC) - 2sum(XiCtYit))/(N*D);

			/*//4. Variance Job: Computes part of variance that requires a distributed job
			*//**
			 * xcty = Sum (xi * C' * yi')
			 *
			 * We also regenerate xi on demand by the following formula:
			 *
			 * xi = yi * y2x
			 *
			 * To make it efficient for sparse matrices that are not mean-centered, we receive the mean
			 * separately:
			 *
			 * xi = (yi - ym) * y2x = yi * y2x - xm, where xm = ym*y2x
			 *
			 * xi * C' * (yi-ym)' = xi * ((yi-ym)*C)' = xi * (yi*C - ym*C)'
			 *
			 *//*

			double ss2 = PCAUtils.trace(centralXtX.times(centralCtC));
			final double[] resArrayYmC = new double[centralC.numCols()];

			final double[] resArrayCYmt = new double[centralCFA.numCols()];

			PCAUtils.denseVectorTimesMatrix(meanVector, centralC, resArrayYmC);
			PCAUtils.denseVectorTimesMatrix(meanVector,centralCFA,resArrayCYmt);

			final Broadcast<Matrix> br_centralC = sc.broadcast(centralC);
			final Broadcast<Matrix> br_centralCFA = sc.broadcast(centralCFA);

			final double[] resArrayYiC = new double[centralC.numCols()];
			final double [] resArrayCYit = new double[centralCFA.numCols()];

			final Accumulator<Double> doubleAccumXctyt = sc.accumulator(0.0);
			final Accumulator <Double> doubleAccumCXYt = sc.accumulator(0.0);

			vectors.foreach( new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {
				public void call(org.apache.spark.mllib.linalg.Vector yi) throws Exception {
					PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2X.value(), resArrayX);
					PCAUtils.sparseVectorTimesMatrix(yi, br_centralC.value(), resArrayYiC );

					PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2Xm.value(), resArrayX_FA);
					PCAUtils.sparseVectorTimesMatrix(yi, br_centralCFA.value(), resArrayCYit);

					PCAUtils.subtract(resArrayYiC, resArrayYmC);
					PCAUtils.subtract(resArrayCYit, resArrayCYmt);

					double dotRes = PCAUtils.dot(resArrayX,resArrayYiC);
					doubleAccumXctyt.add(dotRes);

					double dotRes_FA = PCAUtils.dot(resArrayX_FA, resArrayCYit);
					doubleAccumCXYt.add(dotRes_FA);
				}
			}); //end Variance Job

			double xctyt = doubleAccumXctyt.value();
			double CXYt = doubleAccumCXYt.value();*/



			//back to FA stuff...

			Matrix psiProj = centralCFA.times(centralYProj.transpose());

			psi.viewDiagonal().assign(YCov.viewDiagonal().assign(psiProj.viewDiagonal(),Functions.MINUS));
			psi = psi.divide(nRows); //M Step: new psi calc
			//System.out.print(psi);

			//prevents the numerical error that arises later from taking inv(0) for sparse matrices
			for(int iYCov = 0; iYCov<psi.numRows(); iYCov++){
				if(psi.get(iYCov,iYCov)==0){
					psi.set(iYCov,iYCov,5e-15);
				}
			}

			//System.out.print(psi);


		/*//Compute likelihood


			 CholeskyDecomposition MM = new CholeskyDecomposition(mInv);
			 Matrix L = MM.getL();
			 L.viewDiagonal().assign(Functions.LOGARITHM);
			 double li1 = L.viewDiagonal().aggregate(new DoubleDoubleFunction() {
				 @Override
				 public double apply(double v, double v1) {
					 return v + v1;
				 }
			 }, new DoubleFunction() {
				 @Override
				 public double apply(double v) {
					 return v;
				 }
			 });
*/
			Matrix diag2 = mInv.times(YCovScale);
			//take the trace of diag2
			double li2 = diag2.viewDiagonal().aggregate(new DoubleDoubleFunction() {
				@Override
				public double apply(double v, double v1) {
					return v + v1;
				}
			}, new DoubleFunction() {
				@Override
				public double apply(double v) {
					return v;
				}
			});

			//LL = -nRows/2.0* Math.log(2* Math.PI) + nRows * li1 - nRows * 0.5 * li2; //correct and probably more efficient, but can cause numerical errors
			//log-likelihood
			LL2 = -nRows/2.0* Math.log(2* Math.PI) + nRows * 0.5 * Math.log(mInv.determinant()) - nRows * 0.5 * li2;
			LLChange = LL2 - LLOld;
			LLOld = LL2;

			log.info("...............LOG LIKELIHOOD: " + LL2);
			System.out.println("...............LOG LIKELIHOOD: " + LL2);








			//ss = (norm2 + ss2 - 2 * xctyt) / (nRows * nCols);
			//log.info("SSSSSSSSSSSSSSSSSSSSSSSSSSSS " + ss + " (" + norm2 + " + "
					//+ ss2 + " -2* " + xctyt);


			//double objective = ss;
			//relChangeInObjective = Math.abs(1 - objective / prevObjective);
			//prevObjective = objective;
			//log.info("Objective:  %.6f    relative change: %.6f \n", objective,
					//relChangeInObjective);


			if (!CALCULATE_ERR_ATTHEEND) {
				log.info("Computing the error at round " + round + " ...");
				System.out.println("Computing the error at round " + round + " ...");

				//double[] resArrayZmTmp = new double[centralC.numRows()];

				double [] resArrayZmTmp_FA = new double[centralCFA.numRows()];

				//PCAUtils.vectorTimesMatrixTranspose(xm_mahout, centralC, resArrayZmTmp);
				//final double[] resArrayZm= PCAUtils.subtractVectorArray(resArrayZmTmp , meanVector);

				PCAUtils.vectorTimesMatrixTranspose(xmFA_mahout, centralCFA, resArrayZmTmp_FA);
				final double[] resArrayZm_FA= PCAUtils.subtractVectorArray(resArrayZmTmp_FA, meanVector);

				//5. Reconstruction Error Job: The job computes the reconstruction error of the input matrix after updating the principal
				//components matrix
				/**
				 * Xc = Yc * Y2X
				 *
				 * ReconY = Xc * C'
				 *
				 * Err = ReconY - Yc
				 *
				 * Norm2(Err) = abs(Err).zSum().max()
				 *
				 * To take the sparse matrixes into account we receive the mean separately:
				 *
				 * X = (Y - Ym) * Y2X = X - Xm, where X=Y*Y2X and Xm=Ym*Y2X
				 *
				 * ReconY = (X - Xm) * C' = X*C' - Xm*C'
				 *
				 * Err = X*C' - Xm*C' - (Y - Ym) = X*C' - Y - Zm, where where Zm=Xm*C'-Ym
				 *
				 */

				//final Accumulator<double[]> vectorAccumErr = sc.accumulator(new double[nCols], new VectorAccumulatorAbsParam());
				//final Accumulator<double[]> vectorAccumNormCentralized = sc.accumulator(new double[nCols], new VectorAccumulatorAbsParam());

				final Accumulator<double[]> vectorAccumErr_FA = sc.accumulator(new double[nCols],new VectorAccumulatorAbsParam());
				final Accumulator<double[]> vectorAccumNormCentralized_FA = sc.accumulator(new double[nCols], new VectorAccumulatorParam());

				//final double[] xiCtArray = new double[centralC.numRows()];
				//final double[] normalizedArray = new double[nCols];

				final double[] xiCtArray_FA = new double[centralCFA.numRows()];
				final double [] normalizedArray_FA = new double[nCols];

				final Broadcast<Matrix> br_centralCFA = sc.broadcast(centralCFA);


				vectors.foreach( new VoidFunction<org.apache.spark.mllib.linalg.Vector>() {
					public void call(org.apache.spark.mllib.linalg.Vector yi) throws Exception {
						if (PCAUtils.pass(errRate))
							return;

						//PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2X.value(), resArrayX);
						//PCAUtils.vectorTimesMatrixTranspose(resArrayX, br_centralC.value(), xiCtArray);


						PCAUtils.sparseVectorTimesMatrix(yi, br_centralY2Xm.value(), resArrayX_FA);
						PCAUtils.vectorTimesMatrixTranspose(resArrayX_FA, br_centralCFA.value(), xiCtArray_FA);

						//PCAUtils.denseVectorSubtractSparseSubtractDense(xiCtArray,yi,resArrayZm);
						//vectorAccumErr.add(xiCtArray);

						PCAUtils.denseVectorSubtractSparseSubtractDense(xiCtArray_FA,yi,resArrayZm_FA);
						vectorAccumErr_FA.add(xiCtArray_FA);

						PCAUtils.denseVectorMinusVector(yi, br_ym_mahout.value(), normalizedArray_FA);
						vectorAccumNormCentralized_FA.add(normalizedArray_FA);
					}
				}); // end Reconstruction Error Job

		          /*
		           * The norm of the reconstruction error matrix
		           */
				//double reconstructionError= PCAUtils.getMax(vectorAccumErr.value());
				//log.info("************************ReconsructionError=" + reconstructionError );

				double reconstructionError_FA= PCAUtils.getMax(vectorAccumErr_FA.value());
				log.info("************************ReconsructionError FA=" + reconstructionError_FA );

		          /*
		           * The norm of the input matrix after centralization
		          */
				//double centralizedYNorm= PCAUtils.getMax(vectorAccumNormCentralized.value());
				//log.info("************************CentralizedNOrm=" + centralizedYNorm );

				double centralizedYNorm_FA= PCAUtils.getMax(vectorAccumNormCentralized_FA.value());
				log.info("************************CentralizedNOrm FA=" + centralizedYNorm_FA );

				//error = reconstructionError/centralizedYNorm;
				//log.info("... end of computing the error at round " + round + " And error=" + error);

				error_FA = reconstructionError_FA/centralizedYNorm_FA;
				log.info("... FA end of computing the error at round " + round + " And FA error=" + error_FA);

				//prevError=error;
				prevError_FA=error_FA;
			}
		}
		if(computeProjectedMatrix==1)
		{
			//7. Compute Projected Matrix Job: The job multiplies the input matrix Y with the principal components C to get the Projected vectors
			final Broadcast<Matrix> br_centralC = sc.broadcast(centralC);
			JavaRDD<org.apache.spark.mllib.linalg.Vector> projectedVectors=vectors.map(new Function<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>() {

				public org.apache.spark.mllib.linalg.Vector call(
						org.apache.spark.mllib.linalg.Vector yi) throws Exception {

					return PCAUtils.sparseVectorTimesMatrix(yi,br_centralC.value());
				}
			});//End Compute Projected Matrix
			String path=outputPath+ File.separator + "ProjectedMatrix";
			projectedVectors.saveAsTextFile(path);
		}
		return PCAUtils.convertMahoutToSparkMatrix(centralCFA); //was centralC

	}
	private static void printLogMessage(String argName )
	{
		log.error("Missing arguments -D" + argName);
		log.info("Usage: -i=<path/to/input/matrix> -o=<path/to/outputfolder> -rows=<number of rows> -cols=<number of columns> -pcs=<number of principal components> [-errSampleRate=<Error sampling rate>] [-maxIter=<max iterations>] [-outFmt=<output format>] [-ComputeProjectedMatrix=<0/1 (compute projected matrix or not)>]");
		log.info("Example: spark-submit --class \"org.qcri.sparkpca.SparkPCA\" --master local target/sparkPCA-1.0.jar -i input/seqfiles -o output -rows 7 -cols 5 -pcs 3 -errSampleRate 1");
	}
}
