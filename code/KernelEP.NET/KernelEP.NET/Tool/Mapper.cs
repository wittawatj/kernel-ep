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
			} else if(className.Equals(StackVectorMapper<T1, T2>.MATLAB_CLASS)){
				map = StackVectorMapper<T1, T2>.FromMatlabStruct(s);
			} else if(className.Equals(BayesLinRegFM.MATLAB_CLASS)){
				BayesLinRegFM fm = BayesLinRegFM.FromMatlabStruct(s);
				map = new VectorMapper2Adapter<T1, T2>(fm);
			} else if(className.Equals(UAwareVectorMapper<T1, T2>.MATLAB_CLASS)){
				map = UAwareVectorMapper<T1, T2>.FromMatlabStruct(s);
			}else if(className.Equals(UAwareStackVectorMapper<T1, T2>.MATLAB_CLASS)){
				map = UAwareStackVectorMapper<T1, T2>.FromMatlabStruct(s);
			}else{
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

	public class VectorMapper2Adapter<T1, T2> :  VectorMapper<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		private VectorMapper vecMapper;

		public VectorMapper2Adapter(VectorMapper vecMapper){
			this.vecMapper = vecMapper;
		}

		public override Vector MapToVector(T1 msg1, T2 msg2){
			return vecMapper.MapToVector(new IKEPDist[]{ msg1, msg2 });
		}

		public override int GetOutputDimension(){
			return vecMapper.GetOutputDimension();
		}
	}

	// Correspond to UAwareInstancesMapper in Matlab
	public abstract class UAwareVectorMapper<T1, T2> : VectorMapper<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		public const string MATLAB_CLASS = UAwareVectorMapper.MATLAB_CLASS;
		// Estimate uncertainty on the incoming messages d1 and d2.
		// Uncertainty might be a vector e.g., multioutput operator.
		public abstract double[] EstimateUncertainty(T1 d1, T2 d2);

		// Map and estiamte uncertainty.
		public abstract void MapAndEstimateU(out Vector mapped, 
		                                     out double[] uncertainty, T1 d1, T2 d2);

		public static UAwareVectorMapper<T1, T2> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			UAwareVectorMapper<T1, T2> map = null;
			if(className.Equals(UAwareStackVectorMapper<T1, T2>.MATLAB_CLASS)){
				map = UAwareStackVectorMapper<T1, T2>.FromMatlabStruct(s);
			} else{
				throw new ArgumentException("Unknown className: " + className);
			}

			return map;

		}
	}

	// Correspond to UAwareInstancesMapper in Matlab
	public abstract class UAwareVectorMapper : VectorMapper{
		public const string MATLAB_CLASS = "UAwareInstancesMapper";
		// Estimate uncertainty on the incoming messages d1 and d2.
		// Uncertainty might be a vector e.g., multioutput operator.
		public abstract double[] EstimateUncertainty(params IKEPDist[] dists);

		// Map and estiamte uncertainty.
		public abstract void MapAndEstimateU(out Vector mapped, 
		                                     out double[] uncertainty, params IKEPDist[] dists);

		public static UAwareVectorMapper FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			UAwareVectorMapper map = null;
			if(className.Equals(MATLAB_CLASS)){
				map = null;
			} else{
				throw new ArgumentException("Unknown className: " + className);
			}
		
			return map;

		}
	}

	// same class name in Matlab
	public class BayesLinRegFM : UAwareVectorMapper{
		public const string MATLAB_CLASS = "BayesLinRegFM";

		private VectorMapper featureMap;
		//		% matrix needed in mapInstances(). dz x numFeatures
		//		% where dz = dimension of output sufficient statistic.
		private Matrix mapMatrix;

		//		% posterior covariance matrix. Used for computing predictive variance.
		//		% DxD where D = number of features
		private Matrix posteriorCov;

		//		% output noise variance (regularization parameter)
		private double noise_var;

		private BayesLinRegFM(){
		}

		public override void MapAndEstimateU(out Vector mapped, out double[] uncertainty, 
		                                     params IKEPDist[] msgs){
			Vector feature = featureMap.MapToVector(msgs);
			mapped = mapMatrix * feature;
			double predVar = posteriorCov.QuadraticForm(feature) + noise_var;
			uncertainty = new[]{ predVar };
		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			Vector feature = featureMap.MapToVector(msgs);
			return mapMatrix * feature;
		}

		public override int GetOutputDimension(){
			return mapMatrix.Rows;
		}

		public override int NumInputMessages(){
			return -1;
		}

		public override double[] EstimateUncertainty(params IKEPDist[] dists){
			Vector feature = featureMap.MapToVector(dists);
			double predVar = posteriorCov.QuadraticForm(feature) + noise_var;
			return new double[]{ predVar };
		}

		public static new BayesLinRegFM FromMatlabStruct(MatlabStruct s){
//			s.className=class(this);
//			s.featureMap=this.featureMap.toStruct();
//			%s.regParam=this.regParam;
//			s.mapMatrix=this.mapMatrix;
//			s.posteriorCov = this.posteriorCov; 
//			s.noise_var = this.noise_var;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + MATLAB_CLASS);
			}
			MatlabStruct fmStruct = s.GetStruct("featureMap");
			VectorMapper featureMap = VectorMapper.FromMatlabStruct(fmStruct);
			Matrix mapMatrix = s.GetMatrix("mapMatrix");
			if(mapMatrix.Cols != featureMap.GetOutputDimension()){
				throw new ArgumentException("mapMatrix and featureMap's dimenions are incompatible.");
			}
			Matrix postCov = s.GetMatrix("posteriorCov");
			if(postCov.Cols != featureMap.GetOutputDimension()){
				throw new ArgumentException("posterior covariance and featureMap's dimenions are incompatible.");
			}
			double noise_var = s.GetDouble("noise_var");

			var bayes = new BayesLinRegFM();
			bayes.featureMap = featureMap;
			bayes.mapMatrix = mapMatrix;
			bayes.posteriorCov = postCov;
			bayes.noise_var = noise_var;
			return bayes;
		}
	}

	// Corresponds to FeatureMap in Matlab code.
	// Consider VectorMapper<T1, ...>
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
			} else if(className.Equals(RFGJointKGG.MATLAB_CLASS)){
				map = RFGJointKGG.FromMatlabStruct(s);
			} else if(className.Equals(BayesLinRegFM.MATLAB_CLASS)){
				map = BayesLinRegFM.FromMatlabStruct(s);
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

	// Matlab class = UAwareStackInsMapper
	public class UAwareStackVectorMapper<T1, T2> : UAwareVectorMapper<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		public const string MATLAB_CLASS = UAwareStackVectorMapper.MATLAB_CLASS;
		private readonly UAwareVectorMapper<T1, T2>[] mappers;

		public UAwareStackVectorMapper(params UAwareVectorMapper<T1, T2>[] mappers){
			this.mappers = mappers;
		}

		public override Vector MapToVector(T1 msg1, T2 msg2){
			Vector[] outs = mappers.Select(map => map.MapToVector(msg1, msg2)).ToArray();
			Vector stack = MatrixUtils.ConcatAll(outs);
			return stack;
		}

		public override int GetOutputDimension(){
			return mappers.Sum(map => map.GetOutputDimension());
		}

		public override double[] EstimateUncertainty(T1 d1, T2 d2){
			// ** Take only the fist uncertainty estimate from each mapper.
			double[] U = mappers.Select(map => map.EstimateUncertainty(d1, d2)[0]).ToArray();
			return U;
		}

		public override void MapAndEstimateU(out Vector mapped, 
		                                     out double[] uncertainty, 
		                                     T1 d1, T2 d2){
			int m = mappers.Length;
			uncertainty = new double[m];
			Vector[] outs = new Vector[m];
			for(int i = 0; i < m; i++){
				double[] ui;
				Vector outi;
				mappers[i].MapAndEstimateU(out outi, out ui, d1, d2);
				outs[i] = outi;
				uncertainty[i] = ui[0];
			}
			mapped = MatrixUtils.ConcatAll(outs);
		}

		public new static UAwareStackVectorMapper<T1, T2> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
//			UAwareStackVectorMapper<T1, T2> map = null;
			if(className.Equals(MATLAB_CLASS)){
				UAwareStackVectorMapper svm = UAwareStackVectorMapper.FromMatlabStruct(s);
				return new UAwareStackVectorMapper2Adapter<T1, T2>(svm);
			} else{
				throw new ArgumentException("Unknown className: " + className);
			}


		}
	}

	public class UAwareStackVectorMapper2Adapter<T1, T2>  : UAwareStackVectorMapper<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		private readonly UAwareStackVectorMapper stackMapper;

		public UAwareStackVectorMapper2Adapter(UAwareStackVectorMapper stackMapper){
			this.stackMapper = stackMapper;
		}
		public override Vector MapToVector(T1 msg1, T2 msg2){
			return stackMapper.MapToVector(new IKEPDist[]{msg1, msg2});
		}

		public override int GetOutputDimension(){
			return stackMapper.GetOutputDimension();
		}

		public override double[] EstimateUncertainty(T1 d1, T2 d2){
			return stackMapper.EstimateUncertainty(new IKEPDist[]{d1, d2});
		}

		public override void MapAndEstimateU(out Vector mapped, 
			out double[] uncertainty, T1 d1, T2 d2){
			stackMapper.MapAndEstimateU(out mapped, out uncertainty, d1, d2);
		}

	}

	public class UAwareStackVectorMapper : UAwareVectorMapper{
		public const string MATLAB_CLASS = "UAwareStackInsMapper";
		private readonly UAwareVectorMapper[] mappers;

		public UAwareStackVectorMapper(params UAwareVectorMapper[] mappers){
			this.mappers = mappers;
		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			Vector[] outs = mappers.Select(map => map.MapToVector(msgs)).ToArray();
			Vector stack = MatrixUtils.ConcatAll(outs);
			return stack;
		}

		public override int GetOutputDimension(){
			return mappers.Sum(map => map.GetOutputDimension());
		}

		public override int NumInputMessages(){
			throw new NotImplementedException();
		}

		public override double[] EstimateUncertainty(params IKEPDist[] dists){
			// ** Take only the fist uncertainty estimate from each mapper.
			double[] U = mappers.Select(map => map.EstimateUncertainty(dists)[0]).ToArray();
			return U;
		}

		public override void MapAndEstimateU(out Vector mapped, 
		                                     out double[] uncertainty, params IKEPDist[] dists){
			// ** Take only the fist uncertainty estimate from each mapper.
			int m = mappers.Length;
			uncertainty = new double[m];
			Vector[] outs = new Vector[m];
			for(int i = 0; i < m; i++){
				double[] ui;
				Vector outi;
				mappers[i].MapAndEstimateU(out outi, out ui, dists);
				outs[i] = outi;
				uncertainty[i] = ui[0];
			}
			mapped = MatrixUtils.ConcatAll(outs);
		}

		public new static UAwareStackVectorMapper FromMatlabStruct(MatlabStruct s){
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
			var mappers = new UAwareVectorMapper[mappersCell.GetLength(1)];
			for(int i=0; i<mappers.Length; i++){
				var mapStruct = new MatlabStruct((Dictionary<string, object>)mappersCell[0, i]);
				VectorMapper m1 = VectorMapper.FromMatlabStruct(mapStruct);
				mappers[i] = (UAwareVectorMapper)m1;
			}
			return new UAwareStackVectorMapper(mappers);

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

			var incoming = new Tuple<T1, T2>(msg1,msg2);
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

	// Conceptually the same as UAwareDistMapper in Matlab
	// An uncertainty-aware distribution mapper.
	public interface  IUAwareDistMapper<T, A, B>
		where T : IKEPDist
		where A : IKEPDist
		where B : IKEPDist{

		// Estimate uncertainty on the incoming messages d1 and d2.
		double[] EstimateUncertainty(A d1, B d2);

	}

	// DistMapper based on sufficient statistic vector.
	// T is the type of the target distribution
	//
	// Corresponds to class of the same name in Matlab code
	public class GenericMapper<T, A, B> : DistMapper<T, A, B> 
	where T : IKEPDist where A : IKEPDist where B : IKEPDist{
		// suffMapper maps from messages into sufficient statistic output vector.
		protected VectorMapper<A, B> suffMapper;
		protected DistBuilder<T> distBuilder;
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
			if(!(className.Equals(MATLAB_CLASS)
			   || className.Equals(UAwareGenericMapper<T, A, B>.MATLAB_CLASS))){
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

	// Class with the same name in Matlab.
	// A concrete implementation of an uncertainty-aware distribution mapper.
	public class UAwareGenericMapper<T, A, B>: GenericMapper<T, A, B>, IUAwareDistMapper<T, A, B>
		where T: IKEPDist where A : IKEPDist where B : IKEPDist{

		public new const string MATLAB_CLASS = "UAwareGenericMapper";

		// suffMapper must implement IUAwareVectorMapper
		public UAwareGenericMapper(UAwareVectorMapper<A, B> suffMapper, DistBuilder<T> distBuilder)
			: base(suffMapper, distBuilder){

		}

		public double[] EstimateUncertainty(A d1, B d2){
//			double[] u = vm.EstimateUncertainty(new IKEPDist[]{d1, d2});
//			return u;
			UAwareVectorMapper<A, B> uvm = (UAwareVectorMapper<A, B>)suffMapper;
			double[] u =uvm.EstimateUncertainty(d1, d2);
			return u;
		}


	}
}
// end name space

