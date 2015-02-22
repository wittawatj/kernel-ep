using System;
using System.Diagnostics;
using System.Linq;
using System.Collections.Generic;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;


namespace KernelEP.Tool{

	// abstract class of all kernel functions
	public abstract class Kernel<T>{
		// Evaluate this kernel on the data1 and data2
		public abstract double Eval(T data1, T data2);

		// Return a set of parameters used in this kernel in a cell array.
		// If the kernel does not have any parameter, return an empty cell
		// array, {}.
		//		Param = getParam(this);

		// Evaluate k(x1, y1), k(x2, y2), ....
		public Vector PairEval(List<T> x, List<T> y){
			if(x.Count != y.Count){
				throw new ArgumentException("x and y must have the same length");
			}
			var pair = Enumerable.Range(0, x.Count).Select(i => this.Eval(x[i], y[i]));
			return Vector.FromArray(pair.ToArray());
		}

		public Matrix Eval(List<T> x, List<T> y){
			double[,] gram = new double[x.Count, y.Count];
			for(int i = 0; i < x.Count; i++){
				for(int j = 0; j < y.Count; j++){
					gram[i, j] = Eval(x[i], y[j]);
				}
			}
			return new Matrix(gram);
		}

		public Vector Eval(T x, List<T> y){
			var eval = Enumerable.Range(0, y.Count).Select(i => this.Eval(x, y[i]));
			return Vector.FromArray(eval.ToArray());
		}

		public Vector Eval(List<T> x, T y){
			return Eval(y, x);
		}

		public static Kernel<T> FromMatlabStruct(MatlabStruct s){

			string className = s.GetString("className");
			throw new NotImplementedException();
		}
	}

	// a kernel defined on a stack of 2 inputs
	public abstract class Kernel2<T1, T2> : Kernel<Tuple<T1, T2>>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		public new static Kernel2<T1, T2> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(className.Equals(KDistProduct2<T1, T2>.MATLAB_CLASS)){
				return KDistProduct2<T1, T2>.FromMatlabStruct(s);
			} else{
				throw new ArgumentException("Unknown class in MatlabStruct for Kernel2.");
			}
		}
	
	}

	public class KDistProduct2<T1, T2> : Kernel2<T1, T2>
		where T1 : IKEPDist
		where T2 : IKEPDist{
		private readonly Kernel<T1> ker1;
		private readonly Kernel<T2> ker2;
		public static string MATLAB_CLASS = "KProduct";

		public KDistProduct2(Kernel<T1> ker1, Kernel<T2> ker2){
			this.ker1 = ker1;
			this.ker2 = ker2;
		}

		public override string ToString(){
			return string.Format("[KerProduct2]");
		}

		public override double Eval(Tuple<T1, T2> tup1, Tuple<T1, T2> tup2){
			double k1 = ker1.Eval(tup1.Item1, tup2.Item1);
			double k2 = ker2.Eval(tup1.Item2, tup2.Item2);
			return k1 * k2;
		}

		public new static KDistProduct2<T1, T2> FromMatlabStruct(MatlabStruct s){
//			s = struct();
//			s.className=class(this);
//			kcount = length(this.kernels);
//			kerCell = cell(1, kcount);
//			for i=1:kcount
//					kerCell{i} = this.kernels{i}.toStruct();
//			end
//			s.kernels = kerCell;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(KDistProduct2<T1, T2>));
			}
			MatlabStruct[,] kerStructs = s.GetStructCells("kernels");

			Kernel<T1> k1 = KDist<T1>.FromMatlabStruct(kerStructs[0, 0]);
			Kernel<T2> k2 = KDist<T2>.FromMatlabStruct(kerStructs[0, 1]);
			return new KDistProduct2<T1, T2>(k1,k2);
		}
	}

	public abstract class KDist<T> : Kernel<T> where T : IKEPDist{

		public new static Kernel<T> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");

			if(className.Equals(KEGaussian<T>.MATLAB_CLASS)){
				return KEGaussian<T>.FromMatlabStruct(s);
			}else if(className.Equals(KGGaussian<T>.MATLAB_CLASS)){
				return KGGaussian<T>.FromMatlabStruct(s);
			}else{
				String msg = String.Format("Unknown class: {0}", className);
				throw new ArgumentException(msg);
			}
		}

	}

	/**
	 * Expected product kernel where the mean embeddings are computed with 
	 * a Gaussian kernel.
	*/
	public class KEGaussian<T> : KDist<T> where T : IKEPDist{

		public static string MATLAB_CLASS = "KEGaussian";
		// Gausisan width^2 for each dimension
		private readonly double[] gwidth2s;
		private readonly Matrix sigma;

		// determinant of sigma matrix
		private readonly double detSigma;

		public KEGaussian(double[] gwidth2s){
			this.gwidth2s = gwidth2s;
			sigma = new Matrix(gwidth2s.Length, gwidth2s.Length);
			sigma.SetToDiagonal(Vector.FromArray(gwidth2s));
			detSigma = MatrixUtils.Product(gwidth2s);
			if(detSigma <= 0 || detSigma <= 1e-12){
				string text = string.Format("determinant of Sigma is not proper. Found: {0}.", 
					detSigma);
				throw new ArgumentException(text);
			}
		}

		public override double Eval(T p, T q){
			Vector mp = p.GetMeanVector();
			Vector mq = q.GetMeanVector();
			Matrix covp = p.GetCovarianceMatrix();
			Matrix covq = q.GetCovarianceMatrix();
			// dimensions
			int dp = mp.Count;
			int dq = mq.Count;
			if(dp != dq){
				throw new ArgumentException("dimension mismatch");
			}
			if(dp == 1){
				if(gwidth2s.Length != 1){
					throw new ArgumentException("Expect gwidth2s to have 1 dimension.");
				}
				if( !(covp.Rows ==1 && covp.Cols ==1) ){
					throw new SystemException(String.Format("Except covp to be 1x1. Got {0}x{1}", 
						covp.Rows, covp.Cols));
				}
				if( !(covq.Rows ==1 && covq.Cols ==1) ){
					throw new SystemException(String.Format("Except covq to be 1x1. Got {0}x{1}", 
						covq.Rows, covq.Cols));
				}

				double vp = covp[0, 0];
				double vq = covq[0, 0];
				double kerParam = this.gwidth2s[0];
				double dpq = 1.0/(vp + vq + kerParam);
				double meanDiff = mp[0] - mq[0];
				double exp = Math.Exp(-0.5*meanDiff*dpq*meanDiff);
				double eval = Math.Sqrt(dpq*kerParam)*exp;
				return eval;
			}else{
				// not so efficient. 
				// TODO: improve it
				Matrix dpq = MatrixUtils.Inverse( covp + covq + sigma); 
				Vector meanDiff = mp-mq;
				double dist2 = dpq.QuadraticForm(meanDiff);
				// !! Infer.NET's Determinant() has a bug
				double dpqDet = MatrixUtils.Determinant(dpq);
				double z = Math.Sqrt(dpqDet * detSigma);
				double eval = z*Math.Exp(-0.5*dist2);
				return eval;

			}
		}

		public new static KEGaussian<T> FromMatlabStruct(MatlabStruct s){
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(KEGaussian<T>));
			}
			double[] gwidth2s = s.Get1DDoubleArray("gwidth2s");
			if(!MatrixUtils.IsAllPositive(gwidth2s)){
				throw new ArgumentException("all width^2's must be positive.");
			}
			return new KEGaussian<T>(gwidth2s);
		}
	}

	// Gaussian on mean embeddings as in "Universal kernels on non-standard input
	// spaces" by Christmann and Steinwart.
	public class KGGaussian<T> : KDist<T> where T : IKEPDist{
		public static string MATLAB_CLASS = "KGGaussian";
		// Gaussian width squared for the inner embedding kernel.
		// One width squared for each dimension.
//		private double[] embedSquaredWidths;
		// Expected product kernel for mean embedding
		private KEGaussian<T> keGauss;

		// squred width for the outer Gaussian kernel
		private double squaredWidth;

		public KGGaussian(double[] embedSquaredWidths, double squaredWidth){
//			this.embedSquaredWidths = embedSquaredWidths;
			this.keGauss = new KEGaussian<T>(embedSquaredWidths);
			this.squaredWidth = squaredWidth;
		}


		public override double Eval(T p, T q){
			double dist2 = keGauss.Eval(p, p) - 2*keGauss.Eval(p, q) 
				+ keGauss.Eval(q, q);
			double eval = Math.Exp(-0.5*dist2/squaredWidth);
			return eval;
		}

		public override string ToString(){
			return string.Format("[KGGaussian]");
		}

		/**Compute the median of the pairwise distance. The distance is  
		 * |mu_p - mu_q|^2 where mu_p is the mean embedding.
		*/

		public static double MedianPairwise<D>(
			List<D> dists, double[] embedSquaredWidths)
			where D: IKEPDist{

			KEGaussian<D> ke = new KEGaussian<D>(embedSquaredWidths);
			int n =dists.Count;
			double[] selfKers = dists.Select(di => ke.Eval(di, di)).ToArray();
			List<double> pairDists = new List<double>();
			for(int i=0; i<n; i++){
				D p = dists[i];
				double pp = selfKers[i];
				for(int j=i; j<n; j++){ // include j=i just like in Matlab
					D q = dists[j];
					double qq = selfKers[j];
					double dist2 =  pp - 2*ke.Eval(p, q) + qq;
					Debug.Assert(dist2 >= 0);
					pairDists.Add(dist2);
				}
			}
			double med = MatrixUtils.Median(pairDists.ToArray());
			return med;
		}

		public new static KGGaussian<T> FromMatlabStruct(MatlabStruct s){
//			s = struct();
//			s.className=class(this);
//			s.kegauss = this.kegauss.toStruct();
//			s.embed_width2 = this.embed_width2;
//			s.width2 = this.width2;
//
			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " + typeof(KGGaussian<T>));
			}
//			KEGaussian<T> keGauss = KEGaussian<T>.FromMatlabStruct(s.GetStruct("kegauss"));
			double[] embedSquaredWidths = s.Get1DDoubleArray("embed_width2s");
			if(!MatrixUtils.IsAllPositive(embedSquaredWidths)){
				throw new ArgumentException("all embedding width^2's must be positive.");
			}
			double squaredWidth = s.GetDouble("width2");
			if(squaredWidth <= 0){
				throw new ArgumentException("width2 must be > 0");
			}
			return new KGGaussian<T>(embedSquaredWidths, squaredWidth);
		}

	}
}

