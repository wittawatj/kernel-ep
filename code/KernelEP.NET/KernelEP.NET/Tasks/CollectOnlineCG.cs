using System;
using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using KernelEP.TestIdea;
using KernelEP.Op;
using KernelEP.Tool;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;



/**For running online learning on compound Gamma factor. */
namespace KernelEP{

	public class CollectOnlineCG{

		const int seed_to = 2000;
		const int epIter = 10;
		const double GaussMean = 0;
		const int onlineBatchSizeTrigger = 20;

		public static InferenceEngine NewInferenceEngine(string modelName){
			InferenceEngine ie = new InferenceEngine(new ExpectationPropagation());

		
			//				ie.Compiler.GivePriorityTo(typeof(CGFac4Op));
			// If you want to debug into your generated inference code, you must set this to false:
			ie.Compiler.GenerateInMemory = false;
			ie.Compiler.WriteSourceFiles = true;
			ie.Compiler.GeneratedSourceFolder = Config.PathToCompiledModelFolder();
//			ie.Compiler.GenerateInMemory = true;
//			ie.Compiler.WriteSourceFiles = false;
			ie.Compiler.IncludeDebugInformation = true;
			ie.Algorithm = new ExpectationPropagation();
			ie.NumberOfIterations = epIter;
			//			ie.ShowFactorGraph = true;
			ie.ShowWarnings = true;
			ie.ModelName = modelName;
			ie.ShowTimings = true;
			return ie;

		}

		public static void RunOnlineKEPDotNet(){
			// KJIT with Infer.NET oracle

			// Compile model only once. In this model, only one message can 
			// be collected from one EP problem.
			Variable<int> dataCount = Variable.Observed(-1).Named("dataCount");
			Range n = new Range(dataCount).Named("N");
			Variable<double> precision = Variable<double>.Factor(CGFac.FromCompoundGamma);
			precision.AddAttribute(new TraceMessages());
			precision.AddAttribute(new MarginalPrototype(new Gamma()));
			VariableArray<double> x = Variable.Array<double>(n).Named("X");
			x[n] = Variable.GaussianFromMeanAndPrecision(GaussMean, precision).ForEach(n);


			// Create the  operator instance only once because we want to use the same 
			// one after a new problem (new seed).
			// stopwatch for measuring inference time for each problem
			Stopwatch watch = new Stopwatch();
			CGOpRecords record = new CGOpRecords();
			var opIns = new KEP_CGFacOpIns(onlineBatchSizeTrigger,record,watch);
			opIns.IsPrintTrueWhenCertain = true;
			opIns.IsRecordMessages = true;

			//---- records -----
			List<long> allInferTimes = new List<long>();
			List<long> allOracleInferTimes = new List<long>();
//			List<Dictionary<string, object>> allExtras = new List<Dictionary<string, object>>();
			var allPosteriors = new List<Gamma>();
			var allOraclePosteriors = new List<Gamma>();
			int[] Ns = new int[seed_to];
			double[] trueRate2s = new double[seed_to];
			double[] truePrecs = new double[seed_to];
			List<double[]> allObs = new List<double[]>();
			//---------------

			OpControl.Set(typeof(KEP_CGFacOp), opIns);

			// one engine means the initial model compilation is done only once
			InferenceEngine ie = NewInferenceEngine("KJIT_CG");
			ie.Compiler.GivePriorityTo(typeof(KEP_CGFacOp));

			// engine for the oracle
			InferenceEngine dnetIe = NewInferenceEngine("Oracle_CG");
			dnetIe.Compiler.GivePriorityTo(typeof(CGFacOp));

			for(int seed = 1; seed <= seed_to; seed++){
				int i = seed-1;
				Rand.Restart(seed);
				// n is between 10 and 100 for. Could be different for each seed.
				int N = Rand.Int(10, 100 + 1);
				Console.WriteLine("\n    ///// New compound Gamma problem {0} of size: {1}  /////\n", 
					seed, N);
				Ns[i] = N;
				CompoundGamma cg = new CompoundGamma();

				double[] obs;
				double trueR2, truePrec;
				cg.GenData(N, seed, out obs, out trueR2, out truePrec);
				allObs.Add(obs);
				trueRate2s[i] = trueR2;
				truePrecs[i] = truePrec;

				dataCount.ObservedValue = N;
				x.ObservedValue = obs;

				//				Console.Write("observations: ");
				//				StringUtils.PrintArray(obs);
				//			ie.Compiler.UseParallelForLoops = true;
				watch.Restart();
				Gamma postPrec = ie.Infer<Gamma>(precision);
				long inferenceTime = watch.ElapsedMilliseconds;

				allInferTimes.Add(inferenceTime);
				allPosteriors.Add(postPrec);


//				// Run Infer.net's operator on the same data 
				Stopwatch dnetWatch = new Stopwatch();
//
				dataCount.ObservedValue = N;
				x.ObservedValue = obs;
				dnetWatch.Start();
				Gamma dnetPost = dnetIe.Infer<Gamma>(precision);
				long dnetTime = dnetWatch.ElapsedMilliseconds;
				allOracleInferTimes.Add(dnetTime);
				allOraclePosteriors.Add(dnetPost);


				//print 
				Console.WriteLine("seed: {0}", seed);
				Console.WriteLine("n: {0}", N);
				Console.WriteLine("True r2: {0}", trueR2);
				Console.WriteLine("True precision: {0}", truePrec);
				Console.WriteLine("Inferred precision posterior: {0}", postPrec);
				Console.WriteLine("Infer.NET posterior: {0}", dnetPost);
				Console.WriteLine("Inference time: {0} ms", inferenceTime);
				Console.WriteLine("Infer.NET time: {0} ms", dnetTime);
				Console.WriteLine("=========================");
				Console.WriteLine();
			
			}

			// MatlabWriter cannot write int
			var extra = new Dictionary<string, object>();

			extra.Add("inferTimes",  MatrixUtils.ToDouble(allInferTimes));
			extra.Add("oraInferTimes", MatrixUtils.ToDouble(allOracleInferTimes));
			double[] postShapes, postRates;
			double[] oraPostShapes, oraPostRates;
			CGOpRecords.GammaListToArrays(allPosteriors, out postShapes, out postRates);
			CGOpRecords.GammaListToArrays(allOraclePosteriors, out oraPostShapes, out oraPostRates);
			extra.Add("postShapes", postShapes);
			extra.Add("postRates", postRates);
			extra.Add("oraPostShapes", oraPostShapes);
			extra.Add("oraPostRates", oraPostRates);
			extra.Add("Ns", MatrixUtils.ToDoubleArray(Ns));
			extra.Add("trueRate2s", trueRate2s);
			extra.Add("truePrecs", truePrecs);

			// MatlabWriter cannot write List<Vector> or List<double[]>. Why ?
//			List<Vector> allObsVec = allObs.Select(ob => Vector.FromArray(ob)).ToList();
//			extra.Add("allObs", allObsVec);

			// write the records to a file 

			string fname = string.Format("kjit_cg_iter{0}_bt{1}_st{2}.mat", 
				epIter, onlineBatchSizeTrigger, seed_to);
			string recordPath = Config.PathToSavedFile(fname);
			record.WriteRecords(recordPath, extra);			

		}

	}


}

