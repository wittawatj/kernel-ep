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


namespace KernelEP{
	public class CollectOnlineLogistic{
		const int seed_from = 1;
		const int seed_to = 10;

		const int d = 20;
		const int n = 200;
		const int epIter = 10; 
		const int importanceSamplingSize = 100000;
		const int init_fixed_seed = 1;

		public CollectOnlineLogistic(){ 

		}
		 

		public static void RecordInferNETTime(){
			/**Records time by infer.net*/
		
			/**
			* Only one W just like in Ali's paper. 
			* In practice, we typically observe multiple sets of observations 
			* where we want to do inference on the same model with the same 
			* parameter.
			*/ 
			Rand.Restart(init_fixed_seed);
			Vector w = Vector.Zero(d);
			Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d), w);


			// Create the Logistic operator instance only one because we want to use the same 
			// one after a new problem (new seed).
			// stopwatch for measuring inference time for each problem
			Stopwatch watch = new Stopwatch();
			Type logisticOp = typeof(LogisticOp2);
			LogisticOp2.Watch = watch;
			List<long> allInferTimes = new List<long>();
			var allPosteriors = new List<VectorGaussian>();

			LogisticOp2.IsCollectLogisticMessages = false;
			LogisticOp2.IsCollectProjMsgs = false;
			LogisticOp2.IsCollectXMessages = false;
			for(int seed = seed_from; seed <= seed_to; seed++){

				Rand.Restart(seed);
				double b = 0;
				// combine the bias term into W
				Vector[] X;
				bool[] Y;
				LogisticRegression.GenData(n, w, b, out X, out Y, seed);

				Console.Write("Y: ");
				StringUtils.PrintArray(Y);

				VectorGaussian wPost;

				// start the watch
				watch.Restart();
				LogisticRegression.InferCoefficientsNoBias(X, Y, out wPost, epIter, logisticOp);
				// stop the watch
				long inferenceTime = watch.ElapsedMilliseconds;
				allInferTimes.Add(inferenceTime);

				allPosteriors.Add(wPost);

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
			string fnameM = string.Format("rec_dnet_n{0}_logistic_iter{1}_sf{2}_st{3}.mat", 
				n,  epIter, seed_from, seed_to);
			string recordPathM = Config.PathToSavedFile(fnameM);
			MatlabWriter writer = new MatlabWriter(recordPathM);

			writer.Write("allInferTimes", MatrixUtils.ToDouble(allInferTimes));
			Vector[] postMeans = allPosteriors.Select(vg => vg.GetMean()).ToArray();
			Matrix[] postCovs = allPosteriors.Select(vg => vg.GetVariance()).ToArray();
			writer.Write("postMeans", postMeans);
			writer.Write("postCovs", postCovs);
			writer.Write("dim", d);
			writer.Write("n", n);
			writer.Write("epIter", epIter);
			writer.Write("seed_from", seed_from);
			writer.Write("seed_to", seed_to);
			writer.Write("init_fixed_seed", init_fixed_seed);
			writer.Dispose();
		}

		public static void RunOnlineImportanceSampling(){
			// Run importance sampling for inference

			/**
			 * Only one W just like in Ali's paper. 
			 * In practice, we typically observe multiple sets of observations 
			 * where we want to do inference on the same model with the same 
			 * parameter.
			*/ 
			Rand.Restart(init_fixed_seed);
			Vector w = Vector.Zero(d);
			Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d), w);

			List<LogisticOpRecords> allRecs = new List<LogisticOpRecords>();

			// Create the Logistic operator instance only one because we want to use the same 
			// one after a new problem (new seed).
			// stopwatch for measuring inference time for each problem
			Stopwatch watch = new Stopwatch();
			var logisticOpIns = new ISGaussianLogisticOpIns(
				                    importanceSamplingSize,new LogisticOpRecords(),watch);

			logisticOpIns.IsRecordMessages = true;
			ISGaussianLogisticOp.isLogisticOpIns = logisticOpIns;

			Type logisticOp = typeof(ISGaussianLogisticOp);
			OpControl.Set(logisticOp, logisticOpIns);

			List<long> allInferTimes = new List<long>();
			var allPosteriors = new List<VectorGaussian>();
			var allDotNetPosteriors = new List<VectorGaussian>();

			LogisticOp2.IsCollectLogisticMessages = false;
			LogisticOp2.IsCollectProjMsgs = false;
			LogisticOp2.IsCollectXMessages = false;
			for(int seed = seed_from; seed <= seed_to; seed++){

				Rand.Restart(seed);
				double b = 0;
				// combine the bias term into W
				Vector[] X;
				bool[] Y;
				LogisticRegression.GenData(n, w, b, out X, out Y, seed);

				Console.Write("Y: ");
				StringUtils.PrintArray(Y);

				VectorGaussian wPost;

				LogisticOpRecords recorder = new LogisticOpRecords();
				// Set a new recorder for a new problem seed
				logisticOpIns.records = recorder;
				//			Type logisticOp = typeof(LogisticOp2);

				// start the watch
				watch.Restart();
				LogisticRegression.InferCoefficientsNoBias(X, Y, out wPost, epIter, logisticOp);
				// stop the watch
				long inferenceTime = watch.ElapsedMilliseconds;
				recorder.inferenceTimes = new List<long>();
				recorder.inferenceTimes.Add(inferenceTime);
				allInferTimes.Add(inferenceTime);

				recorder.postW = MatrixUtils.ToList(wPost);
				allPosteriors.Add(wPost);

				allRecs.Add(recorder);
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

				// Run Infer.net's operator on the same data 
				VectorGaussian dotNetPostW;
				LogisticRegression.InferCoefficientsNoBias(X, Y, out dotNetPostW, 
					epIter, typeof(LogisticOp2));
				recorder.dotNetPostW = MatrixUtils.ToList<VectorGaussian>(dotNetPostW);
				allDotNetPosteriors.Add(dotNetPostW);
				// write the records to a file 
				string fname = string.Format("rec_is{0}_n{1}_logistic_iter{2}_s{3}.mat", 
					importanceSamplingSize, n, epIter, seed);
				string recordPath = Config.PathToSavedFile(fname);
				var extra = new Dictionary<string, object>();
				// MatlabWriter cannot write int
				extra.Add("d", (double)d);
				extra.Add("n", (double)n);
				extra.Add("epIter", (double)epIter);
				extra.Add("trueW", w);
				extra.Add("X", MatrixUtils.StackColumns(X));
				extra.Add("Y", MatrixUtils.ToDouble(Y));
				recorder.WriteRecords(recordPath, extra);
			}
			// merge all records and write 
			LogisticOpRecords merged = LogisticOpRecords.Merge(allRecs.ToArray());
			merged.inferenceTimes = allInferTimes;
			merged.dotNetPostW = allDotNetPosteriors;
			merged.postW = allPosteriors;

			string fnameM = string.Format("rec_is{0}_n{1}_logistic_iter{2}_sf{3}_st{4}.mat", 
				importanceSamplingSize, n, epIter, seed_from, seed_to);
			string recordPathM = Config.PathToSavedFile(fnameM);
			merged.WriteRecords(recordPathM);

		}


		public static void RunOnlineKEPSampling(){
			// Kernel EP with importance sampling.
			/**
			 * Only one W just like in Ali's paper. 
			 * In practice, we typically observe multiple sets of observations 
			 * where we want to do inference on the same model with the same 
			 * parameter.
			*/ 
			Rand.Restart(init_fixed_seed);
			Vector w = Vector.Zero(d);
			Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d), w);

			List<LogisticOpRecords> allRecs = new List<LogisticOpRecords>();

			// Create the Logistic operator instance only one because we want to use the same 
			// one after a new problem (new seed).
			// stopwatch for measuring inference time for each problem
			Stopwatch watch = new Stopwatch();
			var logisticOpIns = new KEPOnlineISLogisticOpIns(
				                    new LogisticOpRecords(),watch);

			logisticOpIns.SetImportanceSamplingSize(importanceSamplingSize);
			logisticOpIns.IsRecordMessages = true;
			logisticOpIns.IsPrintTrueWhenCertain = false;

			OpControl.Set(typeof(KEPOnlineLogisticOp), logisticOpIns);
			Type logisticOp = typeof(KEPOnlineLogisticOp);
			 

			List<long> allInferTimes = new List<long>();
			var allPosteriors = new List<VectorGaussian>();
			var allDotNetPosteriors = new List<VectorGaussian>();

			LogisticOp2.IsCollectLogisticMessages = false;
			LogisticOp2.IsCollectProjMsgs = false;
			LogisticOp2.IsCollectXMessages = false;
			for(int seed = seed_from; seed <= seed_to; seed++){

				Rand.Restart(seed);
				double b = 0;
				// combine the bias term into W
				Vector[] X;
				bool[] Y;
				LogisticRegression.GenData(n, w, b, out X, out Y, seed);

				Console.Write("Y: ");
				StringUtils.PrintArray(Y);

				VectorGaussian wPost;

				LogisticOpRecords recorder = new LogisticOpRecords();
				// Set a new recorder for a new problem seed
				logisticOpIns.SetRecorder(recorder);
				//			Type logisticOp = typeof(LogisticOp2);

				// start the watch
				watch.Restart();
				LogisticRegression.InferCoefficientsNoBias(X, Y, out wPost, epIter, logisticOp);
				// stop the watch
				long inferenceTime = watch.ElapsedMilliseconds;
				recorder.inferenceTimes = new List<long>();
				recorder.inferenceTimes.Add(inferenceTime);
				allInferTimes.Add(inferenceTime);

				recorder.postW = MatrixUtils.ToList(wPost);
				allPosteriors.Add(wPost);

				allRecs.Add(recorder);
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

				// Run Infer.net's operator on the same data 
				VectorGaussian dotNetPostW;
				LogisticRegression.InferCoefficientsNoBias(X, Y, out dotNetPostW, 
					epIter, typeof(LogisticOp2));
				recorder.dotNetPostW = MatrixUtils.ToList<VectorGaussian>(dotNetPostW);
				allDotNetPosteriors.Add(dotNetPostW);
				// write the records to a file 
				string fname = string.Format("rec_onlinekep_is{0}_n{1}_logistic_iter{2}_s{3}.mat", 
					importanceSamplingSize, n, epIter, seed);
				string recordPath = Config.PathToSavedFile(fname);
				var extra = new Dictionary<string, object>();
				// MatlabWriter cannot write int
				extra.Add("d", (double)d);
				extra.Add("n", (double)n);
				extra.Add("epIter", (double)epIter);
				extra.Add("trueW", w);
				extra.Add("X", MatrixUtils.StackColumns(X));
				extra.Add("Y", MatrixUtils.ToDouble(Y));
				recorder.WriteRecords(recordPath, extra);
			}
			// merge all records and write 
			LogisticOpRecords merged = LogisticOpRecords.Merge(allRecs.ToArray());
			merged.inferenceTimes = allInferTimes;
			merged.dotNetPostW = allDotNetPosteriors;
			merged.postW = allPosteriors;

			string fnameM = string.Format("rec_onlinekep_is{0}_n{1}_logistic_iter{2}_sf{3}_st{4}.mat", 
				importanceSamplingSize, n, epIter, seed_from, seed_to);
			string recordPathM = Config.PathToSavedFile(fnameM);
			merged.WriteRecords(recordPathM);

		}


	}
}

