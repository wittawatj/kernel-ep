using System;
using System.Collections.Generic;
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
	class MainClass{
		public static void Main(string[] args){
			//Console.WriteLine("Hello World!");
//			SoftmaxOp.Run();
//			TestMultinomialRegression();
//		
//			TestFundamental.TestExpFactor();
//			TestLinq.ArrayFilter1();
//			TestFundamental.TestInferGaussian();
			//TestFundamental.TestMixtureOf2Gaussians();
//			TestReadingMat();
//			TestJaggedSubarray();
			
//			TestMath();

			LogisticRegression.TestLogisticRegression();
//			LogisticRegression.TestLogisticRegressionNoBias();
//			CollectLogisticMsgs.CollectMessages();
//			InferenceEngine.ShowFactorManager(true);


			
//			GauArrayan g1 = new Gaussian(0, 1);
//			Gaussian g2 = new Gaussian(2, 3);
//			Console.WriteLine(g1/g2);
//			Beta b1 = new Beta(1, 2);
//			Beta b2 = new Beta(2, 4);
//			Console.WriteLine(b1/b2);
		
//			Console.WriteLine(DBeta.PointMass(0.3));
//			TestMatrix();
		}

		public static void TestJaggedSubarray(){
			
//			public�static�VariableArray<VariableArray<T>, T[][]> JaggedSubarray<T>(
//	VariableArray<T> array,
//	VariableArray<VariableArray<int>, int[][]> indices
//)
			double[] data = new double[]{ 1, 2, 3, 4, 5, 6 };
			Range r = new Range(data.Length);
			VariableArray<double> varr = Variable.Observed<double>(data);
			
			Range row = new Range(2);
	
			Range col = new Range(3);
			int[][] rawIndex = new int[][] {
				new int[]{ 0, 1, 2 }, new int[]{ 3, 4, 5 }
			};
			VariableArray<VariableArray<int>, int[][]> index = Variable.Observed<int>(rawIndex, row, col).Named("index");
			VariableArray<VariableArray<double>, double[][]> jagged = 
				Variable.JaggedSubarray<double>(varr, index).Named("jagged");
			InferenceEngine ie = new InferenceEngine(new VariationalMessagePassing());
			Console.WriteLine(ie.Infer(jagged));
//			Console.WriteLine(jagged.ObservedValue);
		}

		public static void TestMatrix(){
//			Matrix m =Matrix.IdentityScaledBy(2, 3.0);
			Matrix m = Matrix.Parse("1 2 3\n 4 5 6");
			m[3] = 44;
			Console.WriteLine(m);
			PrintUtils.PrintArray(m.Transpose().SourceArray);

		}

		public static void TestReadingMat(){
			// test reading a .mat file 
//			string path = "/nfs/nhome/live/wittawat/SHARE/gatsby/research/code/saved/test_mat.mat";
			string path = "../../Saved/test_mat1.mat";
			Dictionary<string, Object> dict = MatlabReader.Read(path);
			// a struct is read as a Dictionary<string, Object>
			Console.WriteLine("s: {0}", dict["s"]);
			Dictionary<string, Object> s = (Dictionary<string, Object>)dict["s"];
			Object aobj = s["a"];
			Console.WriteLine("s.a: {0}", aobj);
			Console.WriteLine("s.st: {0}", s["st"]);
			// cell array (1d) is read as Object[,] (2d)
			Object[,] cobj = (Object[,])s["c"];
			Console.WriteLine("s.c: {0}", cobj[0, 1]);
			// a numerical array is read as MicrosoftResearch.Infer.Maths.Matrix
			Object lobj = s["l"];
			Console.WriteLine("s.l: {0}", lobj.GetType());

			// mat file cannot contain Matlab objects. Otherwise, an exception 
			// is thrown.

		}

		public static void TestMultinomialRegression(){
			
			MultinomialRegressionBlog m = new MultinomialRegressionBlog();

			int numSamples = 100;
			int numFeatures = 2;
			int numClasses = 3;
			int countPerSample = 5;
			m.MultinomialRegressionSynthetic(numSamples, numFeatures,
				numClasses, countPerSample);
		}

		public static void TestMath(){
			MMath.Softmax(new double[] { 3.2 });
			IList<Gaussian> g;
		}
	}
}
