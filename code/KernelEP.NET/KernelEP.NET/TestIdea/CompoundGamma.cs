using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Maths;

using MicrosoftResearch.Infer.Models.User;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;
using KernelEP.Op;
using KernelEP.Tool;

/**This file contains code related to a compound gamma factor. */
namespace KernelEP.TestIdea{
	public class CompoundGamma{
		// Nicolas's settings
		public double Rate1 = 3;
		public double Shape1	= 3;
		public double Shape2 = 1;
		public double GaussMean = 0;

		public CompoundGamma(){

		}

		public void GenData(int n, int seed, out double[] data, 
			out double trueR2, out double truePrec){
			Rand.Restart(seed);
			Gamma r2Dist = Gamma.FromShapeAndRate(Shape1, Rate1);
			trueR2 = r2Dist.Sample();
			Gamma precDist = Gamma.FromShapeAndRate(Shape2, trueR2);
			truePrec = precDist.Sample();
			data = new double[n];
			for(int i=0; i<n; i++){
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
			//			ie.Algorithm = new VariationalMessagePassing();
			ie.Algorithm = new ExpectationPropagation();
			//			ie.ModelName = "KEPBinaryLogistic";
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

			for(int seed=1; seed <= 10; seed++){
				Rand.Restart(seed);

				CompoundGamma cg = new CompoundGamma();

				double[] obs;
				double trueR2, truePrec;
				cg.GenData(n, seed, out obs, out trueR2, out truePrec);

				Console.Write("observations: ");
				StringUtils.PrintArray(obs);

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

	}
}

