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

/**This file contains code related to a compound gamma factor. */
namespace KernelEP.Op{
	/**A class containing all information of messages passed/computed in a 
	 * compound gamma message operator.*/
	public class CGOpRecords{

		public List<Gamma> InPrecision = new List<Gamma>();
		public List<Gamma?> OutPrecision = new List<Gamma?>();

		/**Have length = number of messages sent. A true indicates that the oracle 
		is consulted in that time point*/
		public List<bool> ConsultOracle = new List<bool>();
		public List<double[]> Uncertainty = new List<double[]>();

		/**True messages by the oracle.*/
		public List<Gamma> OracleOut = new List<Gamma>();


		public void WriteRecords(string filePath, Dictionary<string, object> extra = null){
		
			MatlabWriter writer = new MatlabWriter(filePath);
			WriteGammaList(writer, InPrecision, "inShape", "inRate");
			WriteGammaList(writer, OutPrecision, "outShape", "outRate");

			// MatlabWritter cannot write boolean
			writer.Write("consultOracle", MatrixUtils.ToDouble(ConsultOracle));
			Matrix uncertaintyMat = MatrixUtils.StackColumnsIgnoreNull(Uncertainty);	

			// write a Matrix
			writer.Write("uncertainty", uncertaintyMat);
			WriteGammaList(writer, OracleOut, "oraOutShape", "oraOutRate");

			if(extra != null){
				foreach(var kv in extra){
					writer.Write(kv.Key, kv.Value);
				}

			}

			writer.Dispose();
		}

		public static void WriteGammaList(MatlabWriter writer, List<Gamma> gList, 
		                                  string shapeName, string rateName){
			double[] shapes, rates;
			GammaListToArrays(gList, out shapes, out rates);
			writer.Write(shapeName, shapes);
			writer.Write(rateName, rates);
		}

		public static void WriteGammaList(MatlabWriter writer, List<Gamma?> gList, 
		                                  string shapeName, string rateName){
			double[] shapes, rates;
			GammaListToArrays(gList, out shapes, out rates);
			writer.Write(shapeName, shapes);
			writer.Write(rateName, rates);
		}

		public void Record(Gamma inPrec, Gamma? outPrec,
		                   bool consult, double[] uncertainty, Gamma oracleOut){
			InPrecision.Add(inPrec);
			OutPrecision.Add(outPrec);
			ConsultOracle.Add(consult);
			Uncertainty.Add(uncertainty);
			OracleOut.Add(oracleOut);

		}

		public static void GammaListToArrays(List<Gamma> l, out double[] shapes, 
		                                     out double[] rates){
			// shape and rate work on improper messages
			int len = l.Count;
			shapes = new double[len];
			rates = new double[len];
			for(int i = 0; i < len; i++){
				shapes[i] = l[i].Shape;
				rates[i] = l[i].Rate;
			}
		}

		public static void GammaListToArrays(List<Gamma?> l, out double[] shapes, 
		                                     out double[] rates){
			// shape and rate work on improper messages
			int len = l.Count;
			shapes = new double[len];
			rates = new double[len];
			for(int i = 0; i < len; i++){
				if(l[i].HasValue){
					shapes[i] = l[i].Value.Shape;
					rates[i] = l[i].Value.Rate;
				} else{
					shapes[i] = double.NaN;
					rates[i] = double.NaN;
				}

			}
		}

		//		public static CGOpRecords Merge(CGOpRecords[] records){
		//			LogisticOpRecords total = new LogisticOpRecords();
		//			for(int i = 0; i < records.Length; i++){
		//				LogisticOpRecords ri = records[i];
		//				total.InLogisticOutLog.AddRange(ri.InLogisticOutLog);
		//
		//			}
		//			return total;
		//
		//		}

	}



	public class CompoundGamma{
		// Nicolas's settings
		public double Rate1 = CGParams.Rate1;
		public double Shape1	= CGParams.Shape1;
		public double Shape2 = CGParams.Shape2;
		public double GaussMean = 0;

		public CompoundGamma(){

		}

		public static double[] DrawFromGaussian(double mean, double precision, int n){
			double[] data = new double[n];
			for(int i = 0; i < n; i++){
				data[i] = Gaussian.Sample(mean, precision);
			}
			return data;
		}

		public void GenData(int n, int seed, out double[] data, 
		                    out double trueR2, out double truePrec){
			Rand.Restart(seed);
			Gamma r2Dist = Gamma.FromShapeAndRate(Shape1, Rate1);
			trueR2 = r2Dist.Sample();
			Gamma precDist = Gamma.FromShapeAndRate(Shape2, trueR2);
			truePrec = precDist.Sample();
			data = new double[n];
			for(int i = 0; i < n; i++){
				Gaussian xDist = Gaussian.FromMeanAndPrecision(GaussMean, truePrec);
				data[i] = xDist.Sample();
			}
		}

		public Gamma InferPrecision(double[] xObs, int epIteration, 
		                            Type gammaOp = null){

			Variable<int> dataCount = Variable.Observed(xObs.Length).Named("dataCount");
			Range n = new Range(dataCount).Named("n");
			Variable<double> rate2 = Variable.GammaFromShapeAndRate(Shape1, Rate1);
			Variable<double> precision = Variable.GammaFromShapeAndRate(Shape2, rate2);
			precision.AddAttribute(new TraceMessages());
			VariableArray<double> x = Variable.Array<double>(n).Named("X");
			x[n] = Variable.GaussianFromMeanAndPrecision(GaussMean, precision)
				.ForEach(n);
			x.ObservedValue = xObs;


			InferenceEngine ie = new InferenceEngine();
			//http://research.microsoft.com/en-us/um/cambridge/projects/infernet/docs/Inference%20engine%20settings.aspx

			if(gammaOp != null){
				ie.Compiler.GivePriorityTo(gammaOp);
			}

			// If you want to debug into your generated inference code, you must set this to false:
			//			ie.Compiler.GenerateInMemory = false;
			//			ie.Compiler.WriteSourceFiles = true;
			//			ie.Compiler.GeneratedSourceFolder = Config.PathToCompiledModelFolder();
			ie.Compiler.GenerateInMemory = true;
			ie.Compiler.WriteSourceFiles = false;

			ie.Compiler.IncludeDebugInformation = true;
			ie.Algorithm = new ExpectationPropagation();
			ie.NumberOfIterations = epIteration;
			//			ie.ShowFactorGraph = true;
			ie.ShowWarnings = true;
			ie.ModelName = "CompoundGamma";
			ie.ShowTimings = true;
			//			ie.Compiler.UseParallelForLoops = true;

			Gamma postPrec = ie.Infer<Gamma>(precision);

			return postPrec;
		}

		public static void TestInference(){
			const int n = 100;
			const int epIter = 10;

			for(int seed = 1; seed <= 5; seed++){
				Rand.Restart(seed);

				CompoundGamma cg = new CompoundGamma();

				double[] obs;
				double trueR2, truePrec;
				cg.GenData(n, seed, out obs, out trueR2, out truePrec);

//				Console.Write("observations: ");
//				StringUtils.PrintArray(obs);

				Gamma postPrec = cg.InferPrecision(obs, epIter);

				//print 
				Console.WriteLine("seed: {0}", seed);
				Console.WriteLine("n: {0}", n);
				Console.WriteLine("True r2: {0}", trueR2);
				Console.WriteLine("True precision: {0}", truePrec);
				Console.WriteLine("Inferred precision posterior: {0}", postPrec);
				Console.WriteLine("=========================");
				Console.WriteLine();
			}


		}

		public Gamma InferPrecisionCustomFactor(double[] xObs, int epIteration, 
		                                        Type gammaOp = null){

			Variable<int> dataCount = Variable.Observed(xObs.Length).Named("dataCount");
			Range n = new Range(dataCount).Named("n");

			Variable<double> precision = Variable<double>.Factor(
				                             CGFac.FromCompoundGamma);
//			Variable<double> precision = Variable<double>.Factor(
//				CGFac4.FromCompoundGamma,
//				CGParams.Shape1,
//				CGParams.Rate1,
//				CGParams.Shape2
//			);

			precision.AddAttribute(new TraceMessages());
			precision.AddAttribute(new MarginalPrototype(new Gamma()));
			VariableArray<double> x = Variable.Array<double>(n).Named("X");
			x[n] = Variable.GaussianFromMeanAndPrecision(GaussMean, precision).ForEach(n);
			x.ObservedValue = xObs;

			InferenceEngine ie = new InferenceEngine();
			//http://research.microsoft.com/en-us/um/cambridge/projects/infernet/docs/Inference%20engine%20settings.aspx

			if(gammaOp != null){
				ie.Compiler.GivePriorityTo(gammaOp);
			} else{
				ie.Compiler.GivePriorityTo(typeof(CGFacOp));
//				ie.Compiler.GivePriorityTo(typeof(CGFac4Op));
			}

			// If you want to debug into your generated inference code, you must set this to false:
			//			ie.Compiler.GenerateInMemory = false;
			//			ie.Compiler.WriteSourceFiles = true;
			//			ie.Compiler.GeneratedSourceFolder = Config.PathToCompiledModelFolder();
			ie.Compiler.GenerateInMemory = false;
			ie.Compiler.WriteSourceFiles = true;

			ie.Compiler.IncludeDebugInformation = true;
			//			ie.Algorithm = new VariationalMessagePassing();
			ie.Algorithm = new ExpectationPropagation();
			//			ie.ModelName = "KEPBinaryLogistic";
			ie.NumberOfIterations = epIteration;
			//			ie.ShowFactorGraph = true;
			ie.ShowWarnings = true;
			ie.ModelName = "CompoundGammaCustom";
			ie.ShowTimings = true;
			//			ie.Compiler.UseParallelForLoops = true;

			Gamma postPrec = ie.Infer<Gamma>(precision);

			return postPrec;
		}

		public static void TestInferenceCustomFactor(){
			const int n = 100;
			const int epIter = 10;

			for(int seed = 1; seed <= 5; seed++){
				Rand.Restart(seed);

				CompoundGamma cg = new CompoundGamma();

				double[] obs;
				double trueR2, truePrec;
				cg.GenData(n, seed, out obs, out trueR2, out truePrec);

//				Console.Write("observations: ");
//				StringUtils.PrintArray(obs);

				Gamma postPrec = cg.InferPrecisionCustomFactor(obs, epIter);

				//print 
				Console.WriteLine("seed: {0}", seed);
				Console.WriteLine("n: {0}", n);
				Console.WriteLine("True r2: {0}", trueR2);
				Console.WriteLine("True precision: {0}", truePrec);
				Console.WriteLine("Inferred precision posterior: {0}", postPrec);
				Console.WriteLine("=========================");
				Console.WriteLine();
			}

		
		}

	}
	// end CompoundGamma class

	/**
	 * A compound Gamma factor to be used with Infer.NET.
	 * Version with 4 variables connected.
	 */
	[Quality(QualityBand.Experimental)]
	public static class CGFac4{

		[Stochastic]
		[ParameterNames("precision","s1","r1","s2")]
		public static double FromCompoundGamma(double s1, double r1, double s2){
//			double s1 = prior.Shape;
//			double r1 = prior.Rate;
			Gamma r2Dist = Gamma.FromShapeAndRate(s1, r1);
			double trueR2 = r2Dist.Sample();
			Gamma precDist = Gamma.FromShapeAndRate(s2, trueR2);
			double precision = precDist.Sample();
			Console.WriteLine("fac4 gen precision: {0}", precision);
			return precision;
		}
	}

	/**Message operator to be used with a CompoundGammaFac4.*/
	[FactorMethod(typeof(CGFac4),"FromCompoundGamma")]
	[Quality(QualityBand.Experimental)]
	public static class CGFac4Op{

		static CGFac4Op(){

		}

		/**I confirmed that the EP using this factor and using 2 Gamma factos
		yields exactly the same result.*/
		// http://research.microsoft.com/en-us/um/cambridge/projects/infernet/docs/How%20to%20add%20a%20new%20factor%20and%20message%20operators.aspx
		public static Gamma PrecisionAverageConditional(Gamma precision, double s1,
		                                                double r1, double s2){

			Console.WriteLine("{0}.PrecisionAverageConditional: precision {1}", 
				typeof(CGFac4Op), precision);
			Gamma r2FwMsg = Gamma.FromShapeAndRate(s1, r1);
			return GammaFromShapeAndRateOp_Slow.SampleAverageConditional(precision,
				s2, r2FwMsg);
		}

	}

	/**
	 * A compound Gamma factor to be used with Infer.NET.
	 * 
	 */
	[Quality(QualityBand.Experimental)]
	public static class CGFac{


		[Stochastic]
		[ParameterNames("precision")]
		public static double FromCompoundGamma(){
			Gamma r2Dist = Gamma.FromShapeAndRate(CGParams.Shape1, 
				               CGParams.Rate1);
			double trueR2 = r2Dist.Sample();
			Gamma precDist = Gamma.FromShapeAndRate(CGParams.Shape2, trueR2);

			double precision = precDist.Sample();
			Console.WriteLine("gen precision: {0}", precision);
			return precision;
		}
	}

	public static class CGParams{
		//		public static double Rate1 = 3;
		//		public static double Shape1	= 3;
		//		public static double Shape2 = 1;
		public static double Rate1 = 1;
		public static double Shape1	= 1;
		public static double Shape2 = 1;
	}

	/**Message operator to be used with a CompoundGammaFac.*/
	[FactorMethod(typeof(CGFac),"FromCompoundGamma")]
	[Quality(QualityBand.Experimental)]
	//[Buffers("r2BwMsg")]
	public static class CGFacOp{
		// this is constant
		private static Gamma r2FwMsg = Gamma.FromShapeAndRate(
			                               CGParams.Shape1, CGParams.Rate1);

		static CGFacOp(){

		}

		/**I confirmed that the EP using this factor and using 2 Gamma factor
		yields exactly the same result.*/
		// http://research.microsoft.com/en-us/um/cambridge/projects/infernet/docs/How%20to%20add%20a%20new%20factor%20and%20message%20operators.aspx
		public static Gamma PrecisionAverageConditional(Gamma precision){
			// This is the default operator.
			Console.WriteLine("{0}.PrecisionAverageConditional: precision {1}", 
				typeof(CGFacOp), precision);
		
			return GammaFromShapeAndRateOp_Slow.SampleAverageConditional(precision,
				CGParams.Shape2, r2FwMsg);
		}

	}

	/**KJIT + oracle operator to be used with a CompoundGammaFac4.*/
	[FactorMethod(typeof(CGFac),"FromCompoundGamma")]
	[Quality(QualityBand.Experimental)]
	public static class KEP_CGFacOp{

		public static Gamma PrecisionAverageConditional(Gamma precision){
			MsgOpInstance opIns = OpControl.Get(typeof(KEP_CGFacOp));
			KEP_CGFacOpIns cgOpIns = (KEP_CGFacOpIns)opIns;
			Gamma toPrec = cgOpIns.PrecisionAverageConditional(precision);
			return toPrec;
		} 

	}

	/**KJIT + oracle operator to be used with a CGFac.
	Learning outgoing messages, not proj messages. Use the true Infer.NET oracle.
	It does not output a proj message.*/
	public class KEP_CGFacOpIns : MsgOpInstance{

		private readonly PrimalGPOnlineMapper<DGamma> toPrecisionMap;

		/**True to record every message passed/computed*/
		public bool IsRecordMessages = true;
		/**True to print the oracle outgoing message even when the opeartor 
		is certain. Slower but necessary for diagnosis.*/
		public bool IsPrintTrueWhenCertain = true;
		// This stop watch measures only the time spent by kernel EP and its consulted oracle
		public Stopwatch watch;
		public CGOpRecords record;

		public  KEP_CGFacOpIns(int onlineBatchSizeTrigger, 
		                       CGOpRecords record = null, Stopwatch watch = null, 
		                       double? unThreshold = null){

			// Ultimately, the uncertainty threshold should be decided automatically.
			// This is for testing purpose.
			this.watch = watch ?? new Stopwatch();
			this.record = record ?? new CGOpRecords();

			BayesLinRegFM toShape = new BayesLinRegFM(RFGJointKGG.EmptyMap());
			toShape.SetOnlineBatchSizeTrigger(onlineBatchSizeTrigger);
			BayesLinRegFM toRate = new BayesLinRegFM(RFGJointKGG.EmptyMap());
			toRate.SetOnlineBatchSizeTrigger(onlineBatchSizeTrigger);
			if(unThreshold != null){
				toShape.SetUncertaintyThreshold(unThreshold.Value);
				toRate.SetUncertaintyThreshold(unThreshold.Value);
			}
			OnlineStackBayesLinReg toPrecSuffMap = new OnlineStackBayesLinReg(toShape,toRate);

			toPrecisionMap = new PrimalGPOnlineMapper<DGamma>(
				toPrecSuffMap,new DGammaLogBuilder());
		}

		public Gamma PrecisionAverageConditional(Gamma precision){
			Console.WriteLine("{0}.PrecisionAverageConditional. precision: {1}",
				typeof(KEP_CGFacOpIns), precision);

			var incom = new IKEPDist[]{ new DGamma(precision) };
			Gamma predictOut;
			// The array should have length 1 as we have one incoming message.
			Vector[] randomFeatures = null;
			watch.Start();
			bool isUn = true;
			bool onlineReady = toPrecisionMap.IsOnlineReady();
			double[] uncertainty = null;
			if(onlineReady){
				randomFeatures = toPrecisionMap.GenAllRandomFeatures(incom);
				Debug.Assert(randomFeatures.Length == 2, 
					"Should have 2 feature vectors: one for shape, one for rate.");
				double[] thresh = toPrecisionMap.GetUncertaintyThreshold();
				uncertainty = toPrecisionMap.EstimateUncertainty(randomFeatures);
				isUn = MatrixUtils.SomeGeq(uncertainty, thresh);
			}
			watch.Stop();

			if(isUn){
				watch.Start();
				// Operator is not certain. Query the oracle.
				Gamma oracleOut = CGFacOp.PrecisionAverageConditional(precision);
				DGamma target = new DGamma(oracleOut);
				// TODO: This line can be improved by passing the random features
				toPrecisionMap.UpdateOperator(target, incom);
				watch.Stop();

				if(IsRecordMessages){
					if(onlineReady){
						// What to do if this is improper ?
						DGamma rawOut = toPrecisionMap.MapToDistFromRandomFeatures(randomFeatures);
						predictOut = (Gamma)rawOut.GetWrappedDistribution();
						double[] logPredVar = uncertainty;
						record.Record(precision, predictOut, true, logPredVar, oracleOut);
					} else{
						// Not record the predicted messages 
						double[] logPredVar = new double[]{ double.NaN, double.NaN };
						record.Record(precision, null, true, logPredVar, oracleOut);
					}
				}
				return oracleOut;
			} else{
				// Operator is sure
				watch.Start();
				DGamma rawOut = toPrecisionMap.MapToDistFromRandomFeatures(randomFeatures);
				predictOut = (Gamma)rawOut.GetWrappedDistribution();
				watch.Stop();

				double[] logPredVar = uncertainty;
				Console.WriteLine(" ** Certain with log predictive variance: {0}", 
					StringUtils.ArrayToString(logPredVar));
				Console.WriteLine("Predicted outgoing: {0}", predictOut);

				Gamma? oracleOut = null;
				if(IsPrintTrueWhenCertain){
					oracleOut = CGFacOp.PrecisionAverageConditional(precision);
					Console.WriteLine("oracle outgoing: {0}", oracleOut);

				}
				Console.WriteLine();

				if(IsRecordMessages){
					// compute oracle's outgoing
					if(oracleOut == null){
						oracleOut = CGFacOp.PrecisionAverageConditional(precision);
					}
					record.Record(precision, predictOut, false, logPredVar, oracleOut.Value);
				}

				return predictOut;
			}

		}

	}




}

