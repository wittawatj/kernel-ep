using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

namespace KernelEP.Tool{

		
	/**A random feature generator*/
	public abstract class RandomFeatureMap : VectorMapper{
		/**
		 * Return the number of random features used.
		 * This is an array because in a multi-staged approximation, we need 
		 * more than one feature vector. For multi-staged approximation, put 
		 * the most outer number of features last.
		*/
		public abstract int[] GetNumFeatures();

		/**
		 * Regenerate this random feature map given a vector of number of features.
		 * This is useful for cross validation with low number of features, 
		 * and increasing it after a good parameter combination is chosen.
		*/
		public abstract RandomFeatureMap Regenerate(int[] numFeatures);


		/**
		 * Generate a list of candidates of this type for parameter selection.
		 * Each candidate uses the specified numFeatures.
		 * medianFactors contains a list of factors to be multipled with the median 
		 * heuristic. The heuristic is different for different random feature maps.
		*/
		public abstract List<RandomFeatureMap> GenCandidates(
			List<IKEPDist[]> msgs, int[] numFeatures, double[] medianFactors, Random rng);


		public static RandomFeatureMap FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			RandomFeatureMap map = null;
			if(className.Equals(RFGJointKGG.MATLAB_CLASS)){
				map = RFGJointKGG.FromMatlabStruct(s);
			} else{
				throw new ArgumentException("Unknown className: " + className);
			}
			//			else if(className.Equals("RFGSumEProdMap")){
			//
			//			}else if(className.Equals("RFGEProdMap")){
			//
			//			}else if(className.Equals("RFGJointEProdMap")){
			//
			//			}else if(className.Equals("RFGProductEProdMap")){
			//
			//			}
			return map;
		}
	}
		

	//	Random Fourier features for Gaussian on mean embeddings
	//using Gaussian kernel (for mean embeddings) on joint distributions.
	public class RFGJointKGG : RandomFeatureMap{
		public const string MATLAB_CLASS = "RFGJointKGG";
		//		% Gaussian widths^2 for the embedding kernel for each incoming variable.
		//		% Length=number of incoming variables.
		//		% One parameter vector for incoming message.
		//		% Reciprocal of gwidth2s are used in drawing W's. Can be a jagged array.
		//		private double[][] embed_width2s_cell;

		//		% width2 for the outer Gaussian kernel on the mean embeddings.
		//		% A scalar.
		private double outer_width2;

		//		% number of random features. This is the same as nfOut i.e., #features
		//		% for the outer map.
		private int numFeatures;

		//		% number of features for the inner map (expected product kernel). An integer.
		private int innerNumFeatures;

		//		% a RFGEProdMap
		private RFGEProdMap eprodMap;

		//		% Din x Dout
		private Matrix Wout = null;

		//	Vector of length numFeatures
		//		% Drawn from U[0, 2*pi]
		private Vector Bout = null;

		private double[] flattenEmbedWidth2s;

		private RFGJointKGG(){}
		/**
		flattenEmbedWidth2s is a concatenation of embed width2 of the 
			 expected product kernel for each input.
		*/
		public RFGJointKGG(double[] flattenEmbedWidth2s, double outer_width2, 
			int innerNumFeatures, int outerNumFeatures){
			if(flattenEmbedWidth2s == null || flattenEmbedWidth2s.Length == 0){
				throw new ArgumentException("embed width2s parameters cannot be empty.");
			}
			if(outer_width2 <= 0){
				throw new ArgumentException("require outer_width2 > 0");
			}
			this.outer_width2 = outer_width2;
			this.numFeatures = outerNumFeatures;
			this.innerNumFeatures = innerNumFeatures;
			this.flattenEmbedWidth2s = flattenEmbedWidth2s;

		}

		public void InitMap(DVectorNormal joint){
			// dynamically initialize the random feature map
			if(Wout== null){
				Debug.Assert(Bout == null);
				int totalDim = joint.GetDimension();
				if(totalDim != flattenEmbedWidth2s.Length){
					throw new SystemException("Expect total dim. of the joint to be = length of params.");
				}

				eprodMap = new RFGEProdMap(flattenEmbedWidth2s, innerNumFeatures);
				Wout = MatrixUtils.Randn(innerNumFeatures, numFeatures)*(1.0/Math.Sqrt(outer_width2));
				double[] Bvec = MatrixUtils.UniformVector(0, 2.0 * Math.PI, numFeatures);
				Bout = Vector.FromArray(Bvec);
				//			VectorGaussian.SampleFromMeanAndVariance();
			}

		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			DVectorNormal joint = ToJointGaussian(msgs);
			InitMap(joint);
			Vector innerFeature = eprodMap.MapToVector(joint);
			Vector FWout = innerFeature * Wout;
			var q = Enumerable.Range(0, numFeatures).Select(i => 
				Math.Cos(FWout[i] + Bout[i]) * Math.Sqrt(2.0 / numFeatures)
			        );
			Vector outerFeature = Vector.FromArray(q.ToArray());
			return outerFeature;
		}

		public override int GetOutputDimension(){
			return numFeatures;
		}

		public override int[] GetNumFeatures(){
			return new[]{ innerNumFeatures, numFeatures };
		}

		public override RandomFeatureMap Regenerate(int[] numFeatures){
			throw new NotImplementedException();
		}

		/**  - Generate a cell array of FeatureMap candidates from medf,
             a list of factors to be  multiplied with the 
             diagonal of the average covariance matrices.

             - subsamples can be used to limit the samples used to compute
             median distance.
             See RFGJointKGG in Matlab
	*/
		public override List<RandomFeatureMap> GenCandidates(
			List<IKEPDist[]> msgs, int[] numFeatures, double[] medianFactors, Random rng){

			if(numFeatures.Length != 2){
				throw new ArgumentException("numFeatures must have length = 2");
			}
			List<DVectorNormal> joints = msgs.Select(m => ToJointGaussian(m)).ToList();
			int n = joints.Count;
			int subsamples = Math.Min(1500, n);
			List<IKEPDist> jointsBase = joints.Cast<IKEPDist>().ToList();
			Vector avgCov = RFGEProdMap.GetAverageDiagCovariance(jointsBase, subsamples, rng);
			double[] embedWidth2 = avgCov.ToArray();

			// generate candidates
			double meanEmbedMedian2 = KGGaussian<DVectorNormal>.MedianPairwise(
				                          joints, embedWidth2);
			var mapCandidates = new List<RandomFeatureMap>(medianFactors.Length);
			for(int i = 0; i < medianFactors.Length; i++){
				double medf = medianFactors[i];
				if(medf <= 0){
					string text = String.Format("medf must be strictly positive. Found i={0}, medf[i]={1}", 
						i, medf);
					throw new ArgumentException(text);
				}
				double width2 = meanEmbedMedian2*medf;
				mapCandidates.Add( new RFGJointKGG(embedWidth2, width2, 
					numFeatures[0], numFeatures[1]) );
			}
			return mapCandidates;
		}

		public static DVectorNormal ToJointGaussian(params IKEPDist[] msgs){
			// Stack all messages to form a big Gaussian distribution
			Vector[] means = msgs.Select(dist => dist.GetMeanVector()).ToArray();
			Matrix[] covs = msgs.Select(dist => dist.GetCovarianceMatrix()).ToArray();
			Vector M = MatrixUtils.ConcatAll(means);
			Matrix V = MatrixUtils.BlkDiag(covs);
			DVectorNormal joint = new DVectorNormal(M,V);
			return joint;
		}

		public new static RFGJointKGG FromMatlabStruct(MatlabStruct s){
			//			s.className=class(this);
			//			s.embed_width2s_cell = this.embed_width2s_cell;
			//			s.outer_width2 = this.outer_width2;
			//			s.numFeatures=this.numFeatures;
			//			s.innerNumFeatures = this.innerNumFeatures;
			//			s.eprodMap=this.eprodMap.toStruct();
			//			s.Wout = this.Wout;
			//			s.Bout = this.Bout;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + MATLAB_CLASS);
			}

			double outer_width2 = s.GetDouble("outer_width2");
			int numFeatures = s.GetInt("numFeatures");
			int innerNumFeatures = s.GetInt("innerNumFeatures");
			MatlabStruct mapStruct = s.GetStruct("eprodMap");
			RFGEProdMap eprodMap = RFGEProdMap.FromMatlabStruct(mapStruct);
			Matrix Wout = s.GetMatrix("Wout");
			if(innerNumFeatures != Wout.Rows){
				throw new ArgumentException("inner #features must be  = #rows of Wout");
			}
			if(numFeatures != Wout.Cols){
				throw new ArgumentException("numFeatures must be = #cols of Wout");
			}
			Vector Bout = s.Get1DVector("Bout");
			if(Bout.Count != numFeatures){
				throw new ArgumentException("Bout must have length = numFeatures");
			}
			RFGJointKGG jointMap = new RFGJointKGG();
			jointMap.outer_width2 = outer_width2;
			jointMap.numFeatures = numFeatures;
			jointMap.innerNumFeatures = innerNumFeatures;
			jointMap.eprodMap = eprodMap;
			jointMap.Wout = Wout;
			jointMap.Bout = Bout;
			// construct object
			return jointMap;
		}

		/**Return a feature map which is not usable. The main reason to have this 
		is for calling GenCandidates.*/
		public static RFGJointKGG EmptyMap(){
			return new RFGJointKGG();
		}
	}

	/** Random Fourier features for expected product kernel using a Gaussian
		kernel for mean embeddings. */
	public class RFGEProdMap : VectorMapper{
		public const string MATLAB_CLASS = "RFGEProdMap";
		//		% Gaussian width^2 for the embedding kernel. Can be a scalar or a vector
		//		% of the same size as the dimension of the input
		private double[] gwidth2;
		//		% number of random features
		private int numFeatures;

		//		% weight matrix. dim x numFeatures
		private Matrix W = null;
		//		% coefficients b. 1 x numFeatures.
		//		% Drawn from U[0, 2*pi]
		private Vector B = null;

		// for manual initialization.
		private RFGEProdMap(){

		}

		public RFGEProdMap(double[] gwidth2, int numFeatures){
			this.gwidth2 = gwidth2;
			this.numFeatures = numFeatures;
			// initialize W, B


		}

		public void InitMap(IKEPDist dist){
			// dynamically initialize the random feature map
			if(W == null){
				Debug.Assert(B == null);

				int dim = dist.GetMeanVector().Count;
				double[] reci = MatrixUtils.Reciprocal(gwidth2);
				double[] zero = Vector.Zero(dim).ToArray();
				W = MatrixUtils.SampleDiagonalVectorGaussian(zero, reci, numFeatures);
				double[] Bvec = MatrixUtils.UniformVector(0, 2.0 * Math.PI, numFeatures);
				B = Vector.FromArray(Bvec);
				//			VectorGaussian.SampleFromMeanAndVariance();
			}

		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			if(msgs.Length != 1){
				string err = string.Format("{0} only works on one distribution", MATLAB_CLASS);
				throw new ArgumentException(err);
			}

			IKEPDist dist = msgs[0];
			InitMap(dist);
			Vector mean = dist.GetMeanVector();
			Matrix cov = dist.GetCovarianceMatrix();
			Vector wtm =  mean * W;
			Matrix VW = cov * W;
			Matrix WVW = new Matrix(W.Rows,W.Cols);
			WVW.SetToElementwiseProduct(VW, W);
			int dim = W.Rows;
			Vector wvwVec = (Vector.Zero(dim) + 1) * WVW;
			double[] cosExp = Enumerable.Range(0, numFeatures).Select(
				                  i => Math.Cos(wtm[i] + B[i]) * Math.Exp(-0.5 * wvwVec[i])
			                  ).ToArray();
			Vector feature = Math.Sqrt(2.0 / numFeatures) * Vector.FromArray(cosExp);
			return feature;

		}

		public override int GetOutputDimension(){
			return numFeatures;
		}

		/**Same implementation in the Matlab code.*/
		public static Matrix GetAverageCovariance(
			List<IKEPDist> dists, int subsamples, Random rng){
			if(subsamples <= 0){
				throw new ArgumentException("Require subsamples > 0");
			}
			if(dists.Count == 0){
				throw new ArgumentException("List of distributions cannot be empty");
			}
			int n = dists.Count;
			int d = dists[0].GetMeanVector().Count;
			List<IKEPDist> subset = MatrixUtils.RandomSubset(dists, Math.Min(n, subsamples), rng);

			// zero matrix
			Matrix sum = Matrix.IdentityScaledBy(d, 0);
			int nsub = subset.Count;
			for(int i = 0; i < nsub; i++){
				sum += subset[i].GetCovarianceMatrix();
			}
			Matrix avg = sum * (1.0 / nsub);
			return avg;

		}

		/**Similar to GetAverageCovariance but compute only the diagonal */
		public static Vector GetAverageDiagCovariance(
			List<IKEPDist> dists, int subsamples, Random rng){

			if(subsamples <= 0){
				throw new ArgumentException("Require subsamples > 0");
			}
			if(dists.Count == 0){
				throw new ArgumentException("List of distributions cannot be empty");
			}
			int n = dists.Count;
			int d = dists[0].GetMeanVector().Count;
			List<IKEPDist> subset = MatrixUtils.RandomSubset(dists, Math.Min(n, subsamples), rng);

			Vector sum = Vector.Zero(d);
			int nsub = subset.Count;
			for(int i = 0; i < nsub; i++){
				sum += subset[i].GetCovarianceMatrix().Diagonal();
			}
			Vector avg = sum * (1.0 / nsub);
			return avg;

		}

		public new static RFGEProdMap FromMatlabStruct(MatlabStruct s){
			//			s.className=class(this);
			//			s.gwidth2=this.gwidth2;
			//			s.numFeatures=this.numFeatures;
			//			s.W=this.W;
			//			s.B=this.B;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + MATLAB_CLASS);
			}

			// Vector of Gaussian width^2 for the mebedding kernel, one for 
			// each dimension of the input. 
			// Can be a scalar i.e., same param for each dimension.
			//			double[] gwidth2 = s.Get1DDoubleArray("gwidth2");
			int numFeatures = s.GetInt("numFeatures");
			// weight matrix. dim x numFeatures.
			Matrix W = s.GetMatrix("W");
			if(W.Cols != numFeatures){
				throw new ArgumentException("numFeatures should be = #cols of W");
			}
			// coefficients b. a vector of length numFeatures. 
			// Drawn form U[0, 2*pi]
			Vector B = s.Get1DVector("B");
			//			int numFeatures = s.GetInt("numFeatures");
			RFGEProdMap map = new RFGEProdMap();
			//			map.gwidth2 = gwidth2;
			map.numFeatures = numFeatures;
			map.W = W;
			map.B = B;
			return map;
		}
	}


	//A finite-dimensional features generator for Euclidean (vector) inputs.
	public abstract class FeatureMap{
		public abstract Vector GenFeatures(Vector input);

		// return the number of features (numFeatures) to be generated.
		public abstract int NumFeatures();

		// return the input dimension expected.
		public abstract int InputDim();
	}

	// Random Fourier Gaussian map as in Rahimi & Recht 2007 for Gaussian kernel.
	// Euclidean (Vector) input version.
	public class RFGMap : FeatureMap{
		// Gaussian width squared
		public double GaussWidthSq { get; private set; }

		// weight matrix. dim x numFeatures
		public Matrix WeightMatrix { get; private set; }
		// vector of uniformly random coefficients b. Length = numFeatures
		public Vector BiasVector { get; private set; }

		// Constructor used in RFGMap.FromDict()
		private RFGMap(){

		}

		// unused at the moment
		private RFGMap(double gaussWidthSq, int numFeatures, int inputDim){
			if(numFeatures <= 0){
				throw new ArgumentException("numFeatures must be positive.");
			}
			if(gaussWidthSq <= 0){
				throw new ArgumentException("Gaussian width must be > 0");
			}
			if(inputDim < 0){
				throw new ArgumentException("inputDim must be > 0");
			}
			// We assume the Gaussian kernel is parameterized by one width.
			this.GaussWidthSq = gaussWidthSq;
			//			this.numFeatures = numFeatures;
			//			this.inputDim = inputDim;

			InitWeights(gaussWidthSq, numFeatures, inputDim);
		}
		//initialize the weightMatrix and biasVector
		private void InitWeights(double gaussWidthSq, int numFeatures, int inputDim){
			// W = weightMatrix should be draw from a Gaussian with variance
			// = 1/gaussWidthSq
			throw new NotImplementedException("don't need it yet. implement later");
			// don't really need this as weightMatrix and biasVector are loaded 
			// anyway. 
		}

		public override int NumFeatures(){
			// W is dim x numFeatures. We output a vector of length numFeatures 
			return WeightMatrix.Cols;
		}

		public override int InputDim(){
			return WeightMatrix.Rows;
		}

		public override Vector GenFeatures(Vector x){
			if(x.Count != InputDim()){
				throw new ArgumentException("Input vector does not have a compatible dimension.");
			}
			// standardize. Divide by sqrt(gauss width^2)
			Vector s = x * (1.0 / Math.Sqrt(this.GaussWidthSq));
			Vector cosArg = WeightMatrix.Transpose() * s + BiasVector;
			double scale = Math.Sqrt(2.0 / NumFeatures());
			var temp = from ca in cosArg
			           select Math.Cos(ca) * scale;
			return Vector.FromArray(temp.ToArray());

		}

		// construct a RFGMap from MatlabStruct.
		// Matlab objects of class RandFourierGaussMap.
		// See RandFourierGaussMap.toStruct()
		public static RFGMap FromMatlabStruct(MatlabStruct s){
			//			s.className = class(this);
			//            s.gwidth2=this.gwidth2;
			//            s.numFeatures=this.numFeatures;
			//            s.dim=this.dim;
			//            s.W=this.W;
			//            s.B=this.B;
			string className = s.GetString("className");
			if(!className.Equals("RandFourierGaussMap")){
				throw new ArgumentException("The input does not represent a " + typeof(RFGMap));
			}

			double gwidth2 = s.GetDouble("gwidth2");
			int numFeatures = s.GetInt("numFeatures");
			//			int dim = s.GetInt("dim");
			Matrix W = s.GetMatrix("W");
			if(W.Rows <= 0 || W.Cols <= 0){
				throw new Exception("Loaded weight matrix has collapsed dimensions");
			}
			if(numFeatures != W.Cols){
				// expect W to be dim x numFeatures
				throw new ArgumentException("Loaded weight matrix's #cols does not match numFeatures.");
			}
			Vector B = s.Get1DVector("B");

			// construct object
			RFGMap map = new RFGMap();
			map.GaussWidthSq = gwidth2;
			map.WeightMatrix = W;
			map.BiasVector = B;

			Console.WriteLine("mapMatrix W's size: ({0}, {1})", W.Rows, W.Cols);
			Console.WriteLine("bias vector length: {0}", B.Count);
			return map;
		}
	}


	// Random Fourier Gaussian map as in Rahimi & Recht for Gaussian kernel.
	// Use mean and variance of a distribution to generate features.
	// - Extract means, variances
	// - Stack them and treat the vector as an input from Euclidean
	// - Put Gaussian kernel on the vector
	// - Approximate the Gaussian kernel with Rahimi & Recht random Fourier features.
	public class RFGMVMap : VectorMapper{
		// Gaussian width^2 for each mean
		public readonly Vector mwidth2s;
		// Gaussian width^2 for each variance.
		public readonly Vector vwidth2s;
		public readonly RFGMap rfgMap;

		public RFGMVMap(Vector mwidth2s, Vector vwidth2s, RFGMap rfgMap){
			if(mwidth2s.Count != vwidth2s.Count){
				throw new ArgumentException("Params. for means and variances must have the same length.");
			}
			for(int i = 0; i < mwidth2s.Count; ++i){
				if(mwidth2s[i] <= 0){
					throw new ArgumentException("mwidth2s: " + mwidth2s + " contains non-positive numbers.");
				}
				if(vwidth2s[i] <= 0){
					throw new ArgumentException("vwidth2s: " + vwidth2s + " contains non-positive numbers.");
				}
			}
			this.mwidth2s = mwidth2s;
			this.vwidth2s = vwidth2s;
			this.rfgMap = rfgMap;
		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			// *** This implementation must match 
			// RandFourierGaussMVMap.genFeatures(.) in Matlab code
			Vector mv = ToMVStack(msgs);
			if(mv.Count != rfgMap.InputDim()){
				throw new ArgumentException("Total MV dimension does not match underlying " + typeof(RFGMap));
			}
			Vector mapped = rfgMap.GenFeatures(mv);
			return mapped;
		}

		public override int GetOutputDimension(){
			return rfgMap.NumFeatures();
		}

		//		public override int NumInputMessages(){
		//			Debug.Assert(mwidth2s.Count == vwidth2s.Count);
		//			return mwidth2s.Count;
		//		}

		private Vector ToMVStack(IKEPDist[] msgs){
			// stack all means and variances. Divide each by its corresponding 
			// sqrt(gauss_width2).
			// *** Implementation must match RandFourierGaussMVMap.toMVStack()
			// in Matlab ***

			// empty vector
			Vector meanStack = Vector.Zero(0);
			Vector varStack = Vector.Zero(0);
			for(int i = 0; i < msgs.Length; i++){
				IKEPDist di = msgs[i];
				Vector smean = di.GetMeanVector() * (1 / Math.Sqrt(mwidth2s[i]));
				Matrix svar = di.GetCovarianceMatrix() * (1 / Math.Sqrt(vwidth2s[i]));
				// reshape column-wise as in Matlab. 
				// Matrix in Infer.NET is stored row-wise
				Vector svarVec = Vector.FromArray(svar.Transpose().SourceArray);

				meanStack = Vector.Concat(meanStack, smean);
				varStack = Vector.Concat(varStack, svarVec);
			}
			Vector mv = Vector.Concat(meanStack, varStack);
			return mv;

		}
		// construct a RFGMVMap from MatlabStruct.
		// Matlab objects of class RadnFourierGaussMVMap.
		// See RandFourierGaussMVMap.toStruct()
		public new static RFGMVMap FromMatlabStruct(MatlabStruct s){
			//            s.className=class(this);
			//            s.mwidth2s=this.mwidth2s;
			//            s.vwidth2s=this.vwidth2s;
			//            s.numFeatures=this.numFeatures;
			//            s.rfgMap=this.rfgMap.toStruct();

			string className = s.GetString("className");
			if(!className.Equals("RandFourierGaussMVMap")){
				throw new ArgumentException("The input does not represent a " + typeof(RFGMVMap));
			}

			// Gaussian width for mean of each input.
			Vector mwidth2s = s.Get1DVector("mwidth2s");
			Vector vwidth2s = s.Get1DVector("vwidth2s");
			//			int numFeatures = s.GetInt("numFeatures");
			RFGMap rfgMap = RFGMap.FromMatlabStruct(s.GetStruct("rfgMap"));

			// construct object
			return new RFGMVMap(mwidth2s,vwidth2s,rfgMap);
		}
	}

}

