using System;
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
	/**
	 * Collect messages from/to a logistic factor in a binary logistic 
	 * regression model.
	*/

	public class CollectLogisticMsgs{
		public CollectLogisticMsgs(){

		}


		public static void CollectMessages(){
			// collect all incoming/outgoing messages from a logistic factor 
			// to its argument x (as in logis = logistic(x) )
			// in a binary logistic regression model. 
			const int seed_from = 1;
			const int seed_to = 20;
			const int d = 10;
			const int n = 400;
			const int epIter = 5;
			// true => collect proj messages instead of outgoing messages
			const bool collectProj = true;
			const string targetAnnotate = collectProj ? "_proj" : "";
			LogisticOp2.IsCollectProjMsgs = collectProj;
			LogisticOp2.IsCollectLogisticMessages = true;
			LogisticOp2.IsCollectXMessages = true; 

			var allToXMsgs = new List<Tuple<Gaussian, Gaussian, Beta>>();
			var allToLogisticMsgs = new List<Tuple<Beta, Gaussian, Beta>>();

			for(int seed = seed_from; seed <= seed_to; seed++){
				Vector w;
				double b;
				Vector[] X;
				bool[] Y;

				Rand.Restart(seed);
				w = Vector.Zero(d);
				Rand.Normal(Vector.Zero(d), PositiveDefiniteMatrix.Identity(d), w);
//				b = Rand.Normal(0, 1);
				b = 0;

				LogisticRegression.GenData(n, w, b, out X, out Y);

				Console.Write("Y: ");
				PrintUtils.PrintArray(Y);

				VectorGaussian wPost;
				Gaussian biasPost;

				LogisticOp2.ResetMessageCollection();
//				LogisticRegression.InferCoefficients(X, Y, out wPost, 
//					out biasPost, epIter, typeof(LogisticOp2));
				LogisticRegression.InferCoefficientsNoBias(X, Y, out wPost, 
					epIter, typeof(LogisticOp2));

				//print 
				Console.WriteLine("n: {0}", n);
				Console.WriteLine("d: {0}", d);
				int t = Y.Sum(o => o ? 1 : 0);
				Console.WriteLine("number of true: {0}", t);
//				Console.WriteLine("True bias: {0}", b);
//				Console.WriteLine("Inferred bias: {0}", biasPost);
				Console.WriteLine("True w: {0}", w);
				Console.WriteLine("Inferred w: ");
				Console.WriteLine(wPost);

				// save the collected messages
				List<Tuple<Gaussian, Gaussian, Beta>> toXMsgs = LogisticOp2.GetToXMessages();
				allToXMsgs.AddRange(toXMsgs);
				string toXFname = string.Format("binlogis_bw{0}_n{1}_iter{2}_s{3}.mat", 
					targetAnnotate, n, epIter, seed);
				string toXPath = Config.PathToSavedFile(toXFname);

				List<Tuple<Beta, Gaussian, Beta>> toLogisticMsgs = LogisticOp2.GetToLogisticMessages();
				allToLogisticMsgs.AddRange(toLogisticMsgs);
				string toLogisticFname = string.Format("binlogis_fw{0}_n{1}_iter{2}_s{3}.mat", 
					targetAnnotate, n, epIter, seed);
				string toLogisticPath = Config.PathToSavedFile(toLogisticFname);

				// extra information apart from collected messages
				var extra = new Dictionary<string, object>() {
					{ "regression_input_dim", (double)d }, 
					{ "true_w", w }, { "true_bias", b }, {
						"regression_training_size",
						(double)n
					}, 
					{ "X", X }, { "Y", Y }
				};
				ToXSerializeToMat(toXPath, toXMsgs, extra);
				ToLogisticSerializeToMat(toLogisticPath, toLogisticMsgs, extra);
			}
			// save combined messages from seed_from to seed_to
			string allToXFname = string.Format("binlogis_bw{0}_n{1}_iter{2}_sf{3}_st{4}.mat", 
				targetAnnotate, n, epIter, seed_from, seed_to);
			string allToXPath = Config.PathToSavedFile(allToXFname);
			ToXSerializeToMat(allToXPath, allToXMsgs, null);

			string allToLogisticFname = string.Format("binlogis_fw{0}_n{1}_iter{2}_sf{3}_st{4}.mat", 
				targetAnnotate, n, epIter, seed_from, seed_to);
			string allToLogisticPath = Config.PathToSavedFile(allToLogisticFname);
			ToLogisticSerializeToMat(allToLogisticPath, allToLogisticMsgs, null);

		}

		public static void ToLogisticSerializeToMat(string file, 
		                                            List<Tuple<Beta, Gaussian, Beta>> msgs, 
		                                            Dictionary<string, object> extra){

			int n = msgs.Count;
			double[] outBetaA = new double[n];
			double[] outBetaB = new double[n];

			double[] inNormalMeans = new double[n];
			double[] inNormalVariances = new double[n];
			// alpha parameters 
			double[] inBetaA = new double[n];
			double[] inBetaB = new double[n];
				
			for(int i = 0; i < msgs.Count; i++){
				Tuple<Beta, Gaussian, Beta> pairI = msgs[i];
				Beta toBeta = pairI.Item1;
				Gaussian fromX = pairI.Item2;
				Beta fromLogistic = pairI.Item3;

				fromX.GetMeanAndVariance(out inNormalMeans[i], out inNormalVariances[i]);
				outBetaA[i] = toBeta.TrueCount;
				outBetaB[i] = toBeta.FalseCount;
				double alpha = fromLogistic.TrueCount;
				double beta = fromLogistic.FalseCount;
				inBetaA[i] = alpha;
				inBetaB[i] = beta;
			}

			// write to .mat file
			MatlabWriter matWriter = new MatlabWriter(file);
			matWriter.Write("outBetaA", outBetaA);
			matWriter.Write("outBetaB", outBetaB);
			matWriter.Write("inNormalMeans", inNormalMeans);
			matWriter.Write("inNormalVariances", inNormalVariances);
			matWriter.Write("inBetaA", inBetaA);
			matWriter.Write("inBetaB", inBetaB);

			if(extra != null){
				foreach(var kv in extra){
					matWriter.Write(kv.Key, kv.Value);
				}
			}
			matWriter.Dispose();
		}


		public static void ToXSerializeToMat(
			string file, List<Tuple<Gaussian, Gaussian, Beta>> msgs, 
			Dictionary<string, object> extra){

			int n = msgs.Count;
			double[] inNormalMeans = new double[n];
			double[] inNormalVariances = new double[n];
			// alpha parameters 
			double[] inBetaA = new double[n];
			double[] inBetaB = new double[n];
			double[] outNormalMeans = new double[n];
			double[] outNormalVariances = new double[n];

			for(int i = 0; i < msgs.Count; i++){
				Tuple<Gaussian, Gaussian, Beta> pairI = msgs[i];
				Gaussian toX = pairI.Item1;
				Gaussian fromX = pairI.Item2;
				Beta fromLogistic = pairI.Item3;

				toX.GetMeanAndVariance(out outNormalMeans[i], out outNormalVariances[i]);
				fromX.GetMeanAndVariance(out inNormalMeans[i], out inNormalVariances[i]);
				double alpha = fromLogistic.TrueCount;
				double beta = fromLogistic.FalseCount;
				inBetaA[i] = alpha;
				inBetaB[i] = beta;
			}

			// write to .mat file
			MatlabWriter matWriter = new MatlabWriter(file);
			matWriter.Write("outNormalMeans", outNormalMeans);
			matWriter.Write("outNormalVariances", outNormalVariances);
			matWriter.Write("inNormalMeans", inNormalMeans);
			matWriter.Write("inNormalVariances", inNormalVariances);
			matWriter.Write("inBetaA", inBetaA);
			matWriter.Write("inBetaB", inBetaB);

			if(extra != null){
				foreach(var kv in extra){
					matWriter.Write(kv.Key, kv.Value);
				}
			}
			matWriter.Dispose();
		}

	}


}

