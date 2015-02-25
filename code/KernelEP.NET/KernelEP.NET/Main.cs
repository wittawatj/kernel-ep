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
using MNMatrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace KernelEP{
	class MainClass{
		public static void Main(string[] args){
//			args = new string[]{"dnet"};
			RunInference(args);
//			Console.WriteLine("{0}", Beta.FromMeanAndVariance(1, 0));
//			TestMultinomialRegression();
//		
			//			SoftmaxOp.Run();
//			TestFundamental.TestExpFactor();
//			TestLinq.ArrayFilter1();
//			TestFundamental.TestInferGaussian();
			//TestFundamental.TestMixtureOf2Gaussians();
//			TestReadingMat();
//			TestJaggedSubarray();
			
//			TestMath();
//			LogisticRegression.TestLogisticRegression();

//			LogisticRegression.TestLogisticRegressionNoBias();
//			CollectOnlineLogistic.RunOnlineKEPSampling();
//			Console.WriteLine(new Matrix(0, 0));

//			CollectLogisticMsgs.CollectMessages();
//			InferenceEngine.ShowFactorManager(true);

			//			Console.WriteLine(MatrixUtils.Randn(30, 5));
//			MatrixUtils.TestStackColumns();
//			GauArrayan g1 = new Gaussian(0, 1);
//			Gaussian g2 = new Gaussian(2, 3);
//			Console.WriteLine(g1/g2);
//			Beta b1 = new Beta(1, 2);
//			Beta b2 = new Beta(2, 4);
//			Console.WriteLine(b1/b2);
//			TestDeterminant();
//			Console.WriteLine(DBeta.PointMass(0.3));
//			TestMatrix();
			 
//			TestImproperMessages();
//			TestWritingMat();
		}

		public static void RunInference(string[] args){
			if(args.Length == 0){
				Console.WriteLine("usage: app.exe {is|kep_is|dnet}");
				Console.WriteLine(" - is for importance sampling.");
				Console.WriteLine(" - kep_is for kernel-based EP with importance sampling oracle.");
				return;
			}
			string routine = args[0];
			if(routine.Equals("is")){
				CollectOnlineLogistic.RunOnlineImportanceSampling();
			}else if(routine.Equals("kep_is")){
				CollectOnlineLogistic.RunOnlineKEPSampling();
			}else if(routine.Equals("dnet")){
				CollectOnlineLogistic.RecordInferNETTime();
			}else{
				string msg = string.Format("unknown routine: {0}", routine);
				throw new ArgumentException(msg);
			}
		}

		public static void TestWritingMat(){
			string p = Config.PathToSavedFile("test_save_mat.mat");
			MatlabWriter w = new MatlabWriter(p);
			w.Write("b", 2);
			double[] arr = new double[]{ 1, 2, 3 };
			w.Write("arr", arr);
			Vector vec = Vector.FromArray(new double[]{ 4, 5, 6, 7 });
			w.Write("vec", vec);

			List<double> list = new List<double>(arr);
			w.Write("list", list);
			Matrix m = Matrix.Parse("1 2\n 3 4");
			w.Write("m", m);

			long time = 1329L;
			w.Write("longNum", time);

			List<Matrix> mats = new List<Matrix>();
			mats.Add(Matrix.IdentityScaledBy(2, 3.0));
			mats.Add(Matrix.IdentityScaledBy(3, 4.0));
			w.Write("list_mats", mats);
			w.Dispose();
		}

		public static void TestImproperMessages(){
			/**
			 * Conclusions: 
			 * - We can get natural parameters from an improper Gaussian 
			 * but not mean and variance.
			 * - We can get true count and false count form an improper Beta 
			 * but not mean and variance.
			*/
//			Gaussian propG = Gaussian.FromMeanAndVariance(0, 1);
			Gaussian imG = Gaussian.FromMeanAndVariance(0, -8);
			double mtp, prec;
			imG.GetNatural(out mtp, out prec);
			Console.WriteLine("mtp: {0}, prec: {1}", mtp, prec);

			Beta imB = Beta.FromMeanAndVariance(0.3, 0.8);
			Console.WriteLine(imB);
			Console.WriteLine(imB.FalseCount);

		}

		public static void TestDeterminant(){
			Matrix m1 = Matrix.Parse("6 0 \n 0 1");
			Matrix m2 = Matrix.Parse("3.0 2.0 \n 1.5 3.0");
			Matrix m3 = Matrix.Parse("0.7493 0.5074\n -0.004 -0.4204");
			Matrix m4 = PositiveDefiniteMatrix.Identity(4);
			Console.WriteLine(m1);
			// expect 6
			Console.WriteLine("m1 det: {0}", m1.Determinant());
			// expect 6
			Console.WriteLine("m2 det: {0}", m2.Determinant());
			// expect -0.313
			Console.WriteLine("m3 det: {0}", m3.Determinant());
			Console.WriteLine("m4 det: {0}", m4.Determinant());
			// expect 6
			Console.WriteLine("m1 my det: {0}", MatrixUtils.Determinant(m1));
			// expect 6
			Console.WriteLine("m2 my det: {0}", MatrixUtils.Determinant(m2));
			// expect -0.313
			Console.WriteLine("m3 my det: {0}", MatrixUtils.Determinant(m3));
			Console.WriteLine("m4 my det: {0}", MatrixUtils.Determinant(m4));

			// Use MathNet
			MNMatrix m5 = MNMatrix.Build.Dense(2, 2, new double[]{ 6, 0, 0, 1 });
			Console.WriteLine(m5);
			Console.WriteLine("m5 det: {0}", m5.Determinant());
		}

		public static void TestMathNetInverse(){
			// http://numerics.mathdotnet.com/Matrix.html
			// inverse is 
//
//			ans =
//
//				0.5000   -0.3333
//				-0.2500    0.5000

			Matrix inferMat = Matrix.Parse("3 2 \n 1.5 3");
			Console.WriteLine(inferMat);
			Matrix inv = MatrixUtils.Inverse(inferMat);
			Console.WriteLine("Inverse: ");
			Console.WriteLine(inv);
		}

		public static void TestMatrixSetTo(){
			double[] vec = { 1, 2, 3, 4, 5, 6 };
			Matrix m = new Matrix(2,3);
			m.SetTo(vec);
			Console.WriteLine(m);
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
			StringUtils.PrintArray(m.Transpose().SourceArray);

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
