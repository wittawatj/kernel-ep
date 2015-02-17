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
	// end of class

	public abstract class VectorMapper<T1, T2> 
		where T1 : IKEPDist
		where T2 : IKEPDist{

		public abstract Vector MapToVector(T1 msg1, T2 msg2);

		// return the dimension of the output mapped vector.
		public abstract int GetOutputDimension();

		public static VectorMapper<T1, T2> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			VectorMapper<T1, T2> map = null;
			if(className.Equals("CondCholFiniteOut")){
				map = CondCholFiniteOut<T1, T2>.FromMatlabStruct(s);
			}else if(className.Equals(StackVectorMapper<T1, T2>.MATLAB_CLASS)){
				map = StackVectorMapper<T1, T2>.FromMatlabStruct(s);
			}  else{
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

	// Corresponds to FeatureMap in Matlab code.
	// Consider VectorMapper<T1, ...>
	[Obsolete]
	public abstract class VectorMapper{
		public abstract Vector MapToVector(params IKEPDist[] msgs);

		// return the dimension of the output mapped vector.
		public abstract int GetOutputDimension();
	
		// return the expected number of incoming messages
		// negative for any number.
		public abstract int NumInputMessages();

		// Load a FeatureMap in Matlab represented by the input
		// All FeatureMap objects can be serialized to struct with .toStruct()
		public static VectorMapper FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			VectorMapper map = null;
			if(className.Equals("RandFourierGaussMVMap")){
				map = RFGMVMap.FromMatlabStruct(s);
			} else if(className.Equals("CondFMFiniteOut")){
				map = CondFMFiniteOut.FromMatlabStruct(s);
			}  else{
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

	// StackInstancesMapper in Matlab code
	public class StackVectorMapper<T1, T2> : VectorMapper<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{

		private readonly VectorMapper<T1, T2> vectorMapper1;
		private readonly VectorMapper<T1, T2> vectorMapper2;
		public const string MATLAB_CLASS = "StackInstancesMapper";

		public StackVectorMapper(VectorMapper<T1, T2> vectorMapper1, 
			VectorMapper<T1, T2> vectorMapper2){
			this.vectorMapper1 = vectorMapper1;
			this.vectorMapper2 = vectorMapper2;
		}

		public override string ToString(){
			return string.Format("[StackVectorMapper]");
		}

		public override int GetOutputDimension(){
			return vectorMapper1.GetOutputDimension() +
			vectorMapper2.GetOutputDimension();
		}


		public override Vector MapToVector(T1 msg1, T2 msg2){
//			if(msgs.Length != 2){
//				throw new ArgumentException("Expect exactly 2 input messages.");
//			}
			Vector mapped1 = vectorMapper1.MapToVector(msg1, msg2);
			Vector mapped2 = vectorMapper2.MapToVector(msg1, msg2);
			return Vector.Concat(mapped1, mapped2);
		}


		public static new StackVectorMapper<T1, T2> FromMatlabStruct(MatlabStruct s){
//			s = struct();
//			s.className=class(this);
//			mapperCount = length(this.instancesMappers);
//			mapperCell = cell(1, mapperCount);
//			for i=1:mapperCount
//					mapperCell{i} = this.instancesMappers{i}.toStruct();
//			end
//			s.instancesMappers = this.instancesMappers;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + MATLAB_CLASS);
			}

			object[,] mappersCell = s.GetCells("instancesMappers");
			if(mappersCell.Length != 2){
				throw new ArgumentException("instancesMappers should have length 2.");
			}
			var map1Struct = new MatlabStruct((Dictionary<string, object>)mappersCell[0, 0]);
			VectorMapper<T1, T2> m1 = VectorMapper<T1, T2>.FromMatlabStruct(map1Struct);
			var map2Struct = new MatlabStruct((Dictionary<string, object>)mappersCell[0, 1]);
			VectorMapper<T1, T2> m2 = VectorMapper<T1, T2>.FromMatlabStruct(map2Struct);
			return new StackVectorMapper<T1, T2>(m1,m2);

		}
	}

	// Corresponds to a class of the same name in Matlab code.
	// Map two input distributions to a finite dimensional vector.
	public class CondCholFiniteOut<T1, T2> : VectorMapper<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		// (Z-Out*R'(RR' + lambda*eye(ra))^-1 R)/lamb. Needed in MapToVector()
		private readonly Matrix zOutR3;

		// input message pairs
		private readonly TensorInstances<T1, T2> inputTensor;

		// kernel function that can operator on the input tensor
		public Kernel2<T1, T2> Kernel { get; private set; }

		public const string MATLAB_CLASS = "CondCholFiniteOut";

		public CondCholFiniteOut(Matrix zOutR3, 
		                         TensorInstances<T1, T2> inputTensor, Kernel2<T1, T2> kernel){
			this.zOutR3 = zOutR3;
			this.inputTensor = inputTensor;
			this.Kernel = kernel;
		}

		public override int GetOutputDimension(){
			return zOutR3.Rows;
		}


		public override Vector MapToVector(T1 msg1, T2 msg2){

			var incoming = new Tuple<T1, T2>(msg1, msg2);
			Vector k = Kernel.Eval(inputTensor.GetAll(), incoming);
			return zOutR3 * k;
		}

		public static new CondCholFiniteOut<T1, T2> FromMatlabStruct(MatlabStruct s){
//			s = struct();
//			s.className=class(this);
//			s.instances = this.In.toStruct();
//			s.kfunc = this.kfunc.toStruct();
//			% a matrix
//			s.ZOutR3 = this.ZOutR3;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " +
				typeof(CondCholFiniteOut<T1, T2>));
			}
			// assume a TensorInstances
//			var instancesDict = (Dictionary<string, object>)s.GetStruct("instances");
			TensorInstances<T1, T2> instances = 
				TensorInstances<T1, T2>.FromMatlabStruct(s.GetStruct("instances"));
			Kernel2<T1, T2> kfunc = Kernel2<T1, T2>.FromMatlabStruct(
				                        s.GetStruct("kfunc"));
			Matrix zOutR3 = s.GetMatrix("ZOutR3");
			return new CondCholFiniteOut<T1, T2>(zOutR3,instances,kfunc);

		}
	}

	// Corresponds to a class of the same name in Matlab code.
	// An InstancesMapper in Matlab code.
	public class CondFMFiniteOut : VectorMapper{
		private VectorMapper featureMap;
		// dz x numFeatures where dz is the dimension of output sufficient
		// statistic
		private Matrix mapMatrix;

		public const string MATLAB_CLASS = "CondFMFiniteOut";

		public CondFMFiniteOut(VectorMapper featureMap, Matrix mapMatrix){
//			Console.WriteLine("featureMap's output dim: {0}", featureMap.GetOutputDimension());
//			Console.WriteLine("mapMatrix's #cols: {0}", mapMatrix.Cols);
			if(featureMap.GetOutputDimension() != mapMatrix.Cols){
				throw new ArgumentException("featureMap output dimension does not match with mapMatrix.");
			}
			this.featureMap = featureMap;
			this.mapMatrix = mapMatrix;
		}

		public override int GetOutputDimension(){
			return mapMatrix.Rows;
		}

		public override int NumInputMessages(){
			// any number of input messages
			return -1;
		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			Vector features = featureMap.MapToVector(msgs);
			return mapMatrix * features;
		}

		public static new CondFMFiniteOut FromMatlabStruct(MatlabStruct s){
//			s.className=class(this);
//            s.featureMap=this.featureMap.toStruct();
//            s.regParam=this.regParam;
//            s.mapMatrix=this.mapMatrix;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(CondFMFiniteOut));
			}

			MatlabStruct mapStruct = s.GetStruct("featureMap");
			VectorMapper featureMap = VectorMapper.FromMatlabStruct(mapStruct);

			// dz x numFeatures where dz is the dimension of output sufficient 
			// statistic
			Matrix mapMatrix = s.GetMatrix("mapMatrix");
			Console.WriteLine("{0}.mapMatrix size: ({1}, {2})", MATLAB_CLASS, 
				mapMatrix.Rows, mapMatrix.Cols);
			return new CondFMFiniteOut(featureMap,mapMatrix);
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

		public override int NumInputMessages(){
			Debug.Assert(mwidth2s.Count == vwidth2s.Count);
			return mwidth2s.Count;
		}

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

	// a map taking incoming messages of types A, B and outputting one
	// output distribution of type T
	//
	// We intentionally parametrize types of incoming messages explicit here as the
	// incoming messages are not arbitrary during the usage of the mapper e.g.,
	// during inference.
	public abstract class DistMapper<T, A, B> 
	where T : IKEPDist
	where A : IKEPDist
	where B : IKEPDist{

		// Map incoming messages to an output message
		public abstract T MapToDist(A msg0, B msg1);

		public static DistMapper<T, A, B> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(className.Equals(GenericMapper<T, A, B>.MATLAB_CLASS)){
				return GenericMapper<T, A, B>.FromMatlabStruct(s);
			} else{
				throw new ArgumentException("Unknown DistMapper to load.");
			}
		}
	}



	// DistMapper based on sufficient statistic vector.
	// T is the type of the target distribution
	//
	// Corresponds to class of the same name in Matlab code
	public class GenericMapper<T, A, B> : DistMapper<T, A, B> 
	where T : IKEPDist where A : IKEPDist where B : IKEPDist{
		// suffMapper maps from messages into sufficient statistic output vector.
		private VectorMapper<A, B> suffMapper;
		private DistBuilder<T> distBuilder;
		public const string MATLAB_CLASS = "GenericMapper";

		public GenericMapper(VectorMapper<A, B> suffMapper, 
		                     DistBuilder<T> distBuilder){
			this.suffMapper = suffMapper;
			this.distBuilder = distBuilder;
		}

		public override T MapToDist(A msg0, B msg1){
			Vector mapped = suffMapper.MapToVector(msg0, msg1);
			// mapped vector and distBuilder must be compatible.
			T outDist = distBuilder.FromStat(mapped);
			return outDist;
		}

		public new static GenericMapper<T, A, B> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals("GenericMapper")){
				throw new ArgumentException("The input does not represent a " +
				typeof(GenericMapper<T, A, B>));
			}
			// nv = number of variables.
			int inVars = s.GetInt("nv");
			if(inVars != 2){
				throw new ArgumentException("Loaded mapper does not expect 2 incoming variables.");
			}
			MatlabStruct rawOperator = s.GetStruct("operator");
			VectorMapper<A, B> instancesMapper = VectorMapper<A, B>.FromMatlabStruct(rawOperator);
			MatlabStruct rawBuilder = s.GetStruct("distBuilder");
			DistBuilder<T> distBuilder = DistBuilderBase.FromMatlabStruct(rawBuilder)
				as DistBuilder<T>;
			return new GenericMapper<T, A, B>(instancesMapper,distBuilder);
		}
	}
}
 // end name space

