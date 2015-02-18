using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Maths;

//using MicrosoftResearch.Infer.Models.User;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;
using KernelEP.Op;
using KernelEP.Tool;


namespace KernelEP.TestIdea{
	public class LogisticRegression{

		public LogisticRegression(){

		}

		public static void GenData(int n, Vector w, double b, 
		                           out Vector[] X, out bool[] Y){

			int d = w.Count;
			X = new Vector[n];
			Y = new bool[n];
			if(w == null){
				throw new ArgumentException("coefficient vector w cannot be null");
			}
//			X = new Vector[n];
//			Y = new double[n];

			for(int i = 0; i < n; i++){
				X[i] = Vector.Zero(d);
				// samples are from standard multivariate normal 
				Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.IdentityScaledBy(d, 1), X[i]);
				// Gamma random noise to each dimension 
//				X[i] = Rand.Gamma(1)*X[i];
		
				double inner = w.Inner(X[i]);
				double p = MMath.Logistic(inner + b);

//				Y[i] = p >= 0.5 ? 1.0 - ep : 0.0 + ep;
				Y[i] = Bernoulli.Sample(p);
//				Y[i] = p >= 0.5;
			}
		}

		public static void InferCoefficientsNoBias(Vector[] xObs, bool[] yObs, 
		                                           out VectorGaussian wPost, int epIteration, 
		                                           Type logisticOperator = null){
			// This is the same binary logistic regression model as in InferCoefficients()
			// without an explicit bias term b. To capture bias, X needs to 
			// be augmented with one extra dimension having a constant value of 1.
			//
			if(logisticOperator == null){
				logisticOperator = typeof(ISLogisticOp);
			}
	
			Variable<int> dataCount = Variable.Observed(xObs.Length).Named("dataCount");
			int D = xObs[0].Count;
			Range n = new Range(dataCount).Named("n");

			// model
			Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(
				                     Vector.Zero(D), PositiveDefiniteMatrix.Identity(D)
			                     ).Named("W");

			//			VariableArray<Vector> x = Variable.Observed<Vector>(null, n).Named("X");
			VariableArray<Vector> x = Variable.Observed<Vector>(xObs, n).Named("X");
			VariableArray<double> logisticArgs = Variable.Array<double>(n).Named("logist_arg");
			logisticArgs[n] = Variable.InnerProduct(x[n], w).Named("inner");

			//			VariableArray<bool> yData = Variable.Observed<bool>(null, n).Named("Y");
			VariableArray<bool> yData = Variable.Array<bool>(n).Named("Y");
			VariableArray<double> probs = Variable.Array<double>(n).Named("probs");
			probs[n] = Variable.Logistic(logisticArgs[n]).Named("logistic");
			yData[n] = Variable.Bernoulli(probs[n]);
			// set observed values
			x.ObservedValue = xObs;
			yData.ObservedValue = yObs;

			InferenceEngine ie = new InferenceEngine();
			ie.Compiler.GivePriorityTo(logisticOperator);
			ie.Compiler.GenerateInMemory = true;
			//			ie.Algorithm = new VariationalMessagePassing();
			ie.Algorithm = new ExpectationPropagation();
			//			ie.ModelName = "KEPBinaryLogistic";
			ie.NumberOfIterations = epIteration;
			//			ie.ShowFactorGraph = true;
			ie.ShowWarnings = true;
//			ie.Compiler.UseParallelForLoops = true;
			wPost = ie.Infer<VectorGaussian>(w);
		}


		public static void InferCoefficients(
			Vector[] xObs, bool[] yObs, out VectorGaussian wPost, 
			out Gaussian biasPost, int epIteration, 
			Type logisticOperator = null){
    
			if(logisticOperator == null){
				logisticOperator = typeof(ISLogisticOp);
			}
			// yObs expected to be an array of 0, 1

			// number of classes = 2
//			int N = xObs.Length;
			Variable<int> dataCount = Variable.Observed(xObs.Length).Named("dataCount");
			int D = xObs[0].Count;
			Range n = new Range(dataCount).Named("n");

			// model
//			Vector meanW = Vector.FromArray(2, 3);
			Vector meanW = Vector.Zero(D);
			Variable<Vector> w = Variable.VectorGaussianFromMeanAndPrecision(
				                     meanW, PositiveDefiniteMatrix.Identity(D)).Named("W");

			// bias term is normally distributed.
			Variable<double> bias = Variable.GaussianFromMeanAndPrecision(0, 1)
				.Named("bias");

//			VariableArray<Vector> x = Variable.Observed<Vector>(null, n).Named("X");
			VariableArray<Vector> x = Variable.Observed<Vector>(xObs, n).Named("X");
			VariableArray<double> logisticArgs = Variable.Array<double>(n).Named("logist_arg");
			logisticArgs[n] = Variable.InnerProduct(x[n], w).Named("inner") + bias;

			VariableArray<bool> yData = Variable.Observed<bool>(null, n).Named("Y");
//			VariableArray<bool> yData = Variable.Array<bool>(n).Named("Y");
			VariableArray<double> probs = Variable.Array<double>(n).Named("probs");
			probs[n] = Variable.Logistic(logisticArgs[n]).Named("logistic");
			yData[n] = Variable.Bernoulli(probs[n]);
			// set observed values
			x.ObservedValue = xObs;
			yData.ObservedValue = yObs;

			// initialization
//			w.InitialiseTo(VectorGaussian.FromMeanAndPrecision(
//				Vector.FromArray(Rand.Normal(), Rand.Normal()),
//				PositiveDefiniteMatrix.Identity(D)));
//			bias.InitialiseTo(Gaussian.FromMeanAndVariance(Rand.Normal(), 10));
//			yData[n] = logisticArgs[n];

			// inference
			InferenceEngine ie = new InferenceEngine();

//			ie.Algorithm = new VariationalMessagePassing();
			ie.Algorithm = new ExpectationPropagation();
//			ie.ModelName = "KEPBinaryLogistic";
			ie.NumberOfIterations = epIteration;
//			ie.ShowFactorGraph = true;
			ie.ShowWarnings = true;
//			ie.ShowTimings = true;
			ie.Compiler.AddComments = true;
//			ie.Compiler.GivePriorityTo(typeof(MicrosoftResearch.Infer.Factors.LogisticOp));
//			ie.Compiler.UseSerialSchedules = true;
//			ie.Compiler.GivePriorityTo(typeof(KEPLogisticOp));
			KEPLogisticOp.SetPrintTrueMessages(true);
			ie.Compiler.GivePriorityTo(logisticOperator);
//			ie.Compiler.GivePriorityTo(typeof(ISLogisticOp));
//			ie.Compiler.GivePriorityTo(typeof(KernelEP.Op.LogisticOp2));
//			ie.Compiler.GivePriorityTo(typeof(KernelEP.Op.ISLogisticOp));

			ie.Compiler.GenerateInMemory = true;
//			ie.Compiler.UseParallelForLoops = true;
//			ie.Compiler.GeneratedSourceFolder = "/nfs/nhome/live/wittawat/SHARE/gatsby/research2/KernelEP.NET/KernelEP.NET/Compiled";
//			ie.Compiler.WriteSourceFiles = true;
//			ie.Compiler.IncludeDebugInformation = true;		
	
			
//			 load FactorOperator
//			string factorOpPath = Config.PathToFactorOperator("serialFactorOp_ichol_n400_iter5_sf1_st200_ntr4000.mat");


			// get compiled algorithm
//			IGeneratedAlgorithm ca = ie.GetCompiledInferenceAlgorithm(w, bias);
//			KEPBinaryLogistic_EP ca = new KEPBinaryLogistic_EP();
//			ca.dataCount = xObs.Length;
//			ca.X = xObs;
//			ca.Y = yObs;
//			ca.SetObservedValue("dataCount", xObs.Length);
//			ca.SetObservedValue("X", xObs);
//			ca.SetObservedValue("Y", yObs);
//			ca.Execute(1);
//			wPost = ca.Marginal<VectorGaussian>("W");
//			biasPost = ca.Marginal<Gaussian>("bias");
			
			wPost = ie.Infer<VectorGaussian>(w);
			biasPost = ie.Infer<Gaussian>(bias);
		}

		public static LogisticOpParams LoadLogisticFactorOperator(string filePath){
			// filePath to .mat file containing a serialized FactorOperator from 
			// Matlab.
			// These files are typicall in saved/factor_op/
			Dictionary<string, object> dict = MatlabReader.Read(filePath);
			Dictionary<string, object> s = dict["serialFactorOp"] as Dictionary<string, object>;
			LogisticOpParams factorOp = LogisticOpParams.FromMatlabStruct(new MatlabStruct(s));
			return factorOp;
		}


		public static void TestLogisticRegressionNoBias(){
			const int seed = 2;
			Rand.Restart(seed);
			const int d = 10;
			const int n = 400;
			const int epIter = 10;

			Vector w = Vector.Zero(d);
			Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d), w);
			double b = 0;
			// combine the bias term into W
			Vector[] X;
			bool[] Y;
			GenData(n, w, b, out X, out Y);

			Console.Write("Y: ");
			PrintUtils.PrintArray(Y);

			VectorGaussian wPost;

			string factorOpPath = Config.PathToFactorOperator(
				//				"serialFactorOp_fm_kgg_joint_irf500_orf1000_n400_iter5_sf1_st20_ntr5000.mat"
				"serialFactorOp_fm_kgg_joint_irf500_orf1000_proj_n400_iter5_sf1_st20_ntr5000.mat"
			);
			LogisticOpParams factorOp = LoadLogisticFactorOperator(factorOpPath);
			OpControl.Add(typeof(KEPLogisticOp), factorOp);
			LogisticOp2.IsCollectXMessages = false;
			LogisticOp2.IsCollectLogisticMessages = false;
			//			Type logisticOp = typeof(KEPLogisticOp);
			Type logisticOp = typeof(LogisticOp2);

			InferCoefficientsNoBias(X, Y, out wPost, epIter, logisticOp);

			//print 
			Console.WriteLine("n: {0}", n);
			Console.WriteLine("d: {0}", d);
			int t = Y.Sum(o => o ? 1 : 0);
			Console.WriteLine("number of true: {0}", t);
			Console.WriteLine("True bias: {0}", b);
//			Vector meanW = wPost.GetMean();

			Console.WriteLine("True w: {0}", w);
			Console.WriteLine("Inferred w: ");
			Console.WriteLine(wPost);

		}

		public static void TestLogisticRegression(){
			const int seed = 39;
			Rand.Restart(seed);
			const int d = 10;
			const int n = 100;
			const int epIter = 10;

			Vector w = Vector.Zero(d);
			Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d), w);
			double b = Rand.Normal(0, 1);
//			double b= 0;
			Vector[] X;
			bool[] Y;
			GenData(n, w, b, out X, out Y);

			Console.Write("Y: ");
			PrintUtils.PrintArray(Y);

			VectorGaussian wPost;
			Gaussian biasPost;

			Type logisticOp = typeof(KEPLogisticOp);
//			Type logisticOp = typeof(LogisticOp2);

			//			string factorOpPath = Config.PathToFactorOperator("ver7.mat");
			//			string factorOpPath = Config.PathToFactorOperator("factorOp_RFGMVMapperLearner_nicolas_sigmoid_25000.mat");
			//			string factorOpPath = Config.PathToFactorOperator("serialFactorOp_ichol_logbeta_n400_iter5_sf1_st200_ntr6000.mat");
			string factorOpPath = Config.PathToFactorOperator(
//				"serialFactorOp_fm_kgg_joint_irf500_orf1000_n400_iter5_sf1_st20_ntr5000.mat"
				                      "serialFactorOp_fm_kgg_joint_irf500_orf1000_proj_n400_iter5_sf1_st20_ntr5000.mat"
			                      );
			LogisticOpParams factorOp = LoadLogisticFactorOperator(factorOpPath);
			OpControl.Add(typeof(KEPLogisticOp), factorOp);
			LogisticOp2.IsCollectXMessages = false;
			LogisticOp2.IsCollectLogisticMessages = false;

			InferCoefficients(X, Y, out wPost, out biasPost, epIter, logisticOp);

			//print 
			Console.WriteLine("n: {0}", n);
			Console.WriteLine("d: {0}", d);
			int t = Y.Sum(o => o ? 1 : 0);
			Console.WriteLine("number of true: {0}", t);
			Console.WriteLine("True bias: {0}", b);
			Console.WriteLine("Inferred bias: {0}", biasPost);
			Console.WriteLine("True w: {0}", w);
			Console.WriteLine("Inferred w: ");
			Console.WriteLine(wPost);

		}


	}

}

