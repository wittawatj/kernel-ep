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
	// Random Fourier features for expected product kernel using a Gaussian
	// ernel for mean embeddings.
	public class RFGEProdMap : VectorMapper{
		public const string MATLAB_CLASS = "RFGEProdMap";
		//		% Gaussian width^2 for the embedding kernel. Can be a scalar or a vector
		//		% of the same size as the dimension of the input distribution.
		//		private double[] gwidth2;
		//		% number of random features
		private int numFeatures;

		//		% weight matrix. dim x numFeatures
		private Matrix W;
		//		% coefficients b. 1 x numFeatures.
		//		% Drawn from U[0, 2*pi]
		private Vector B;

		// for manual initialization.
		private RFGEProdMap(){

		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			if(msgs.Length != 1){
				string err = string.Format("{0} only works on one distribution", MATLAB_CLASS);
				throw new ArgumentException(err);
			}

			IKEPDist dist = msgs[0];
			Vector mean = dist.GetMeanVector();
			Matrix cov = dist.GetCovarianceMatrix();
			Vector wtm = mean * W;
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

	//	Random Fourier features for Gaussian on mean embeddings
	//using Gaussian kernel (for mean embeddings) on joint distributions.
	public class RFGJointKGG : VectorMapper{
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
		private Matrix Wout;

		//	Vector of length numFeatures
		//		% Drawn from U[0, 2*pi]
		private Vector Bout;

		private RFGJointKGG(){
		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			Vector[] means = msgs.Select(dist => dist.GetMeanVector()).ToArray();
			Matrix[] covs = msgs.Select(dist => dist.GetCovarianceMatrix()).ToArray();
			Vector M = MatrixUtils.ConcatAll(means);
			Matrix V = MatrixUtils.BlkDiag(covs);
			DVectorNormal joint = new DVectorNormal(M,V);
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
			if(Bout.Count() != numFeatures){
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
	}

}

