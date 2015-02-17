using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

namespace KernelEP.TestIdea{
	public class TestFundamental{
		public static void TestExpFactor(){
			//int n = 100;
			//Range range = new Range(n).Named("n");
			Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
			Variable<double> prec = Variable.GammaFromShapeAndScale(1, 1).Named("variance");
			Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, prec).Named("x");
			//VariableArray<double> a = Variable<double>.Array(new Range(1));

			Variable<double> z = Variable.Exp(-x).Named("z");
			z.ObservedValue = 4;
			InferenceEngine ie = new InferenceEngine();
			//ie.Algorithm = new VariationalMessagePassing();
			ie.ShowFactorGraph = true;
			Console.WriteLine(ie.Infer(x));
		}

		public static void TestInferGaussian(){
			// Infer a 1d Gaussian parameters from some observed data
			
			// sample data from Gaussian
			double[] data = new double[100];
			double trueMean = 1;
			double trueVariance = 1;
			for(int i=0; i<data.Length; i++){
				data[i] = Rand.Normal(trueMean, Math.Sqrt(trueVariance));
			}


			// some Gaussian with unknown parameters
			Range range = new Range(data.Length).Named("n");
			VariableArray<double> x = Variable.Array<double>(range).Named("x");
			Variable<double> mean = Variable.GaussianFromMeanAndPrecision(0, 0.1).Named("mean");
			//Variable<double> variance = Variable.GammaFromShapeAndScale(2, 3).Named("variance");
			Variable<double> prec = Variable.GammaFromShapeAndScale(1, 0.1).Named("prec");
			x[range] = Variable.GaussianFromMeanAndPrecision(mean, prec).ForEach(range);
			x.ObservedValue = data;

			// inference
			InferenceEngine engine = new InferenceEngine();
			//engine.Algorithm = new VariationalMessagePassing();
//			engine.ShowFactorGraph = true;

			// output mean as a Gaussian with mean and precision
			Console.WriteLine("mean = " + engine.Infer(mean));

			// output Gamma (with shape and scale)
			Console.WriteLine("prec = " + engine.Infer(prec));
		}

		public static void TestMixtureOf2Gaussians(){
			int mixtureSize = 2;
			Range r = new Range(mixtureSize);
			Variable<int> comp = Variable<int>.Discrete(r, new double[]{0.5, 0.5});
			Variable<double> x = Variable<double>.New<double>().Named("x");
			double[] means = new double[]{1, 2};
			VariableArray<double> Means = Variable.Observed(means, r);
			using(Variable.Switch(comp)){
				x.SetTo(Variable.GaussianFromMeanAndPrecision(Means[comp], 1));
			}
			x.AddAttribute(new TraceMessages());

			InferenceEngine engine = new InferenceEngine();
			engine.OptimiseForVariables = new[]{x};
			engine.Compiler.IncludeDebugInformation = true;
			engine.Algorithm = new GibbsSampling();
			engine.ModelName = "MyMixtureOf2Gaussians";
			engine.ShowTimings = true;
//			engine.ShowMsl = true;
//			InferenceEngine.ShowFactorManager(true);
			// this gives a Gaussian with mean 1.5
			Console.WriteLine("inferred x = " + engine.Infer(x));
			// How to get a Gaussian mixture ????
			// => put prior on means and infer the the distribution on means instead 

		}
		
		
		public static void Test1DRegression(){


		}
		
		public static void Test(){
	


		}
	}

}

