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
		const int seed_to = 50;

		const int d = 20;
		const int n = 300;
		const int epIter = 10;
		const int importanceSamplingSize = 500000;
		const int init_fixed_seed = 1;

		/**for real data experiment*/
		string[] dataNames;
		string[] dataAbbrv;
		string[] dataPaths;


		public CollectOnlineLogistic(){ 
			dataNames = new string[]{"banknote_norm_tr.mat", "blood_transfusion_norm_tr.mat", 
				"ionosphere_norm_tr.mat", "fertility_norm_tr.mat"
			};
			dataAbbrv = new string[]{ "banknote", "blood", "iono", "fertility" };
			dataPaths  = dataNames.Select(dn => Config.PathToSavedFile("data/" + dn)).ToArray();
			Debug.Assert(dataNames.Length == dataAbbrv.Length);
			Debug.Assert(dataAbbrv.Length == dataPaths.Length);

		}


		public  void RecordInferNETTime(){
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
				                n, epIter, seed_from, seed_to);
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

		public  void RunOnlineImportanceSampling(){
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
			merged.WriteRecords(recordPathM );

		}

		public  void LoadDataFromMat(string filePath, out Vector[] X, out bool[] Y){
			// Expect X (dxn) and Y (1xn) in the mat file 
			Dictionary<string, object> dict = MatlabReader.Read(filePath);
			Matrix xmat = (Matrix)dict["X"];
			X = MatrixUtils.SplitColumns(xmat);
			int n = X.Length;
			// Y is actually a 1xn vector
			Matrix ymat = (Matrix)dict["Y"];
			double[,] yarr = ymat.ToArray();
			if(yarr.GetLength(1) != n){
				throw new ArgumentException("length of X and Y do not match.");
			}
			Y = new bool[n];
			for(int i = 0; i < n; i++){
				// expect 0, 1
				Y[i] = yarr[0, i] > 0.5;
			}

		}

		/**Run inference with an importance sampler  on a number of UCI real datasets.*/
		public  void RunRealOnlineSampling(){
			Rand.Restart(1);

			List<LogisticOpRecords> allRecs = new List<LogisticOpRecords>();

			// Create the Logistic operator instance only once because we want to use the same 
			// one after a new problem (new seed).
			// stopwatch for measuring inference time for each problem
			Stopwatch watch = new Stopwatch();
			var logisticOpIns = new ISGaussianLogisticOpIns(
				importanceSamplingSize, new LogisticOpRecords(), watch);
			logisticOpIns.IsRecordMessages = true;
			ISGaussianLogisticOp.isLogisticOpIns = logisticOpIns;

			Type logisticOp = typeof(ISGaussianLogisticOp);
			OpControl.Set(logisticOp, logisticOpIns);

			List<long> allInferTimes = new List<long>();
			List<long> allOraInferTimes = new List<long>();
			var allPosteriors = new List<VectorGaussian>();
			var allDotNetPosteriors = new List<VectorGaussian>();

			LogisticOp2.IsCollectLogisticMessages = false;
			LogisticOp2.IsCollectProjMsgs = false;
			LogisticOp2.IsCollectXMessages = false;
			string folder = "online_uci/";
			for(int i = 0; i < dataNames.Length; i++){
				Console.WriteLine();
				Console.WriteLine("----------- starting problem {0} --------------", dataAbbrv[i]);
				Console.WriteLine();

				Vector[] X;
				bool[] Y;
				LoadDataFromMat(dataPaths[i], out X, out Y);
				Console.Write("Y: ");
				StringUtils.PrintArray(Y);

				VectorGaussian wPost;
				LogisticOpRecords recorder = new LogisticOpRecords();
				// Set a new recorder for a new problem seed
				logisticOpIns.records = recorder;
				// start the watch
				watch.Restart();
				// We do not include the bias term. So make sure the datasets 
				// are standardized.
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

				//			Vector meanW = wPost.GetMean();

				Console.WriteLine("Inferred w: ");
				Console.WriteLine(wPost);

				// Run Infer.net's operator on the same data 
				VectorGaussian dotNetPostW;
				Stopwatch oraWatch = new Stopwatch();
				oraWatch.Start();
				LogisticRegression.InferCoefficientsNoBias(X, Y, out dotNetPostW, 
					epIter, typeof(LogisticOp2));
				long oraInferTime = oraWatch.ElapsedMilliseconds;
				allOraInferTimes.Add(oraInferTime);
				recorder.dotNetPostW = MatrixUtils.ToList<VectorGaussian>(dotNetPostW);
				allDotNetPosteriors.Add(dotNetPostW);


				// write the records to a file 
				string fname = string.Format("is{0}_{1}_iter{2}.mat", 
					importanceSamplingSize, dataAbbrv[i], epIter);
				string recordPath = Config.PathToSavedFile(folder + fname);
				var extra = new Dictionary<string, object>();
				// MatlabWriter cannot write int
				extra.Add("d", (double)X[0].Count);
				extra.Add("n", (double)X.Length);
				extra.Add("epIter", (double)epIter);

				extra.Add("X", MatrixUtils.StackColumns(X));
				extra.Add("Y", MatrixUtils.ToDouble(Y));
				recorder.WriteRecords(recordPath, extra);
			}
			// merge all records and write 
			LogisticOpRecords merged = LogisticOpRecords.Merge(allRecs.ToArray());
			merged.inferenceTimes = allInferTimes;
			merged.dotNetPostW = allDotNetPosteriors;
			merged.postW = allPosteriors;

			string fnameM = string.Format("is{0}_uci{1}_iter{2}.mat", 
				importanceSamplingSize, dataAbbrv.Length, epIter);
			string recordPathM = Config.PathToSavedFile(folder + fnameM);
			var allExtra = new Dictionary<string, object>();
			double[] oraInferTimesArr = allOraInferTimes.Select(t => (double)t).ToArray();
			allExtra.Add("oraInferTimes", Vector.FromArray(oraInferTimesArr));
			merged.WriteRecords(recordPathM, allExtra);
		}


		/**Run KJIT with an importance sampler as the oracle on a number of 
		UCI real datasets.*/
		public  void RunRealOnlineKEPSampling(){
			Rand.Restart(1);

			List<LogisticOpRecords> allRecs = new List<LogisticOpRecords>();

			// Create the Logistic operator instance only once because we want to use the same 
			// one after a new problem (new seed).
			// stopwatch for measuring inference time for each problem
			Stopwatch watch = new Stopwatch();
			var logisticOpIns = new KEPOnlineISLogisticOpIns(
				new LogisticOpRecords(),watch, -8.95);

			logisticOpIns.SetOnlineBatchSizeTrigger(500);
			logisticOpIns.SetImportanceSamplingSize(importanceSamplingSize);
			logisticOpIns.IsRecordMessages = true;
			logisticOpIns.IsPrintTrueWhenCertain = false;

			// See BayesLinRegFM's  BatchLearn() and KEPOnlineISLogisticOpIns()
			// If using the sum kernel
			logisticOpIns.SetFeatures(new int[]{ 400, 800 });

			OpControl.Set(typeof(KEPOnlineLogisticOp), logisticOpIns);
			Type logisticOp = typeof(KEPOnlineLogisticOp);

			List<long> allInferTimes = new List<long>();
			List<long> allOraInferTimes = new List<long>();
			var allPosteriors = new List<VectorGaussian>();
			var allDotNetPosteriors = new List<VectorGaussian>();

			LogisticOp2.IsCollectLogisticMessages = false;
			LogisticOp2.IsCollectProjMsgs = false;
			LogisticOp2.IsCollectXMessages = false;
			string folder = "online_uci/";
			for(int i = 0; i < dataNames.Length; i++){
				Console.WriteLine();
				Console.WriteLine("----------- starting problem {0} --------------", dataAbbrv[i]);
				Console.WriteLine();

				Vector[] X;
				bool[] Y;
				LoadDataFromMat(dataPaths[i], out X, out Y);
				Console.Write("Y: ");
				StringUtils.PrintArray(Y);

				VectorGaussian wPost;
				LogisticOpRecords recorder = new LogisticOpRecords();
				// Set a new recorder for a new problem seed
				logisticOpIns.SetRecorder(recorder);
				//			Type logisticOp = typeof(LogisticOp2);

				// start the watch
				watch.Restart();
				// We do not include the bias term. So make sure the datasets 
				// are standardized.
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

				//			Vector meanW = wPost.GetMean();

				Console.WriteLine("Inferred w: ");
				Console.WriteLine(wPost);

				// Run Infer.net's operator on the same data 
				VectorGaussian dotNetPostW;
				Stopwatch oraWatch = new Stopwatch();
				oraWatch.Start();
				LogisticRegression.InferCoefficientsNoBias(X, Y, out dotNetPostW, 
					epIter, typeof(LogisticOp2));
				long oraInferTime = oraWatch.ElapsedMilliseconds;
				allOraInferTimes.Add(oraInferTime);
				recorder.dotNetPostW = MatrixUtils.ToList<VectorGaussian>(dotNetPostW);
				allDotNetPosteriors.Add(dotNetPostW);


				// write the records to a file 
				string fname = string.Format("kjit_is{0}_{1}_iter{2}.mat", 
					               importanceSamplingSize, dataAbbrv[i], epIter);
				string recordPath = Config.PathToSavedFile(folder + fname);
				var extra = new Dictionary<string, object>();
				// MatlabWriter cannot write int
				extra.Add("d", (double)X[0].Count);
				extra.Add("n", (double)X.Length);
				extra.Add("epIter", (double)epIter);

				extra.Add("X", MatrixUtils.StackColumns(X));
				extra.Add("Y", MatrixUtils.ToDouble(Y));
				recorder.WriteRecords(recordPath, extra);
			}
			// merge all records and write 
			LogisticOpRecords merged = LogisticOpRecords.Merge(allRecs.ToArray());
			merged.inferenceTimes = allInferTimes;
			merged.dotNetPostW = allDotNetPosteriors;
			merged.postW = allPosteriors;

			string fnameM = string.Format("kjit_is{0}_uci{1}_iter{2}.mat", 
				                importanceSamplingSize, dataAbbrv.Length, epIter);
			string recordPathM = Config.PathToSavedFile(folder + fnameM);
			var allExtra = new Dictionary<string, object>();
			double[] oraInferTimesArr = allOraInferTimes.Select(t => (double)t).ToArray();
			allExtra.Add("oraInferTimes", Vector.FromArray(oraInferTimesArr));
			merged.WriteRecords(recordPathM, allExtra);
		}


		public  void RunOnlineKEPSampling(){
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
				new LogisticOpRecords(),watch, -8.5);

			logisticOpIns.SetImportanceSamplingSize(importanceSamplingSize);
			logisticOpIns.IsRecordMessages = true;
			logisticOpIns.IsPrintTrueWhenCertain = false;
			/** Use mixture or not ...*/
			logisticOpIns.isGaussianOp.useMixtureProposal = false;
			logisticOpIns.SetFeatures(new int[]{300, 500});

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

