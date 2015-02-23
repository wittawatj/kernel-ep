using System;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using MNMatrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;


namespace KernelEP{
	public static class MatrixUtils{


		/**
		 * Find an inverse of Infer.NET's matrix. 
		 * Infer.NET does not implement Inverse() even though the method is 
		 * there.
		*/
		public static Matrix Inverse(Matrix m){
			// http://numerics.mathdotnet.com/Matrix.html
			if(m.Rows != m.Cols){
				throw new ArgumentException("input matrix must be square");
			}
			MNMatrix mnMat = ToMathNetMatrix(m);
			MNMatrix mnInv = mnMat.Inverse();
			double[,] invArr = mnInv.ToArray();
			return new Matrix(invArr);
		}
		public static List<T> ToList<T>(params T[] arr){
			return arr.ToList();
		}
		/**Convert from Infer.NET's Matrix to MathNet's Matrix*/
		public static MNMatrix ToMathNetMatrix(Matrix m){
			double[,] arr = m.ToArray();
			// convert to MathNet's matrix
			var MBuild = MNMatrix.Build;
			MNMatrix mnMat = MBuild.DenseOfArray(arr);
			return mnMat;
		}

		public static List<double> ToDouble(List<bool> list){
			// convert to a list of 0-1 double list 
			return list.Select(v => v ? 1.0 : 0).ToList();
		}
		public static List<double> ToDouble(bool[] list){
			// convert to a list of 0-1 double list 
			return list.Select(v => v ? 1.0 : 0).ToList();
		}
		public static List<double> ToDouble(List<int> list){
			// convert to a list of 0-1 double list 
			return list.Select(v => (double)v).ToList();
		}


		public static bool IsAllPositive(double[] nums){
			// True of if all elements are > 0
			double[] filtered = nums.Where(x => x > 0).ToArray();
			return filtered.Length == nums.Length;
		}

		public static Matrix BlkDiag(params Matrix[] mats){
			// Block-diagonally stack all the matrices
			int n = mats.Length;
			int totalRows = mats.Sum(m => m.Rows);
			int totalCols = mats.Sum(m => m.Cols);
			Matrix big = new Matrix(totalRows,totalCols);
			int firstRow = 0, firstCol = 0;
			for(int i = 0; i < n; i++){
				big.SetSubmatrix(firstRow, firstCol, mats[i]);
				firstRow += mats[i].Rows;
				firstCol += mats[i].Cols;
			}
			return big;
		}
		//---------
		public static void TestBlkDiag(){
			Matrix m1 = Matrix.Parse("1 2\n 3 4");
			Matrix m2 = Matrix.Parse("5 6 7\n 8 9 10 ");
			Matrix m3 = Matrix.Parse("11\n12");
			Matrix m4 = Matrix.Parse("13 14");
			Matrix big = MatrixUtils.BlkDiag(m1, m2, m3, m4);
			Console.WriteLine(big);
		}

		public static Vector ConcatAll(params Vector[] vecs){
			// Stack all vectors together
			Vector big = Vector.Zero(0);
			foreach(Vector v in vecs){
				big = Vector.Concat(big, v);
			}
			return big;
		}

		public static bool Or(bool[] vs){
			// boolean or 
			foreach(bool v in vs){
				if(v){
					return true;
				}
			}
			return false;
		}

		public static bool And(bool[] vs){
			// boolean or 
			foreach(bool v in vs){
				if(!v){
					return false;
				}
			}
			return true;
		}

		public static double[] Reciprocal(double[] vec){
			var q = vec.Select(v => 1.0 / v);
			return q.ToArray();
		}

		/**Like in Matlab*/
		public static int[] Randperm(int n, int k, Random rng){
			int[] ind = Enumerable.Range(0, n).ToArray();
			Shuffle<int>(ind, rng);
			return ind.Take(k).ToArray();
		}

		public static T[] RandomSubset<T>(T[] array, int k, Random rng){
			// take a random subset of size k without replacement.
			int n = array.Length;
			int[] kInd = Randperm(n, k, rng);
			T[] subset = Enumerable.Range(0, k).Select(i => array[kInd[i]]).ToArray();
			return subset;
		}

		public static List<T> RandomSubset<T>(List<T> list, int k, Random rng){
			// take a random subset of size k without replacement.
			int n = list.Count;
			int[] kInd = Randperm(n, k, rng);
			List<T> subset = Enumerable.Range(0, k).Select(i => list[kInd[i]]).ToList();
			return subset;
		}

		public static void Shuffle<T>(T[] array, Random rng){
			//http://stackoverflow.com/questions/108819/best-way-to-randomize-a-string-array-with-net
			int n = array.Length;
			while(n > 1){
				int k = rng.Next(n--);
				T temp = array[n];
				array[n] = array[k];
				array[k] = temp;
			}
		}

		/**
		 * Sample from a Gaussian with diagonal covariance.
		 * diag is the diagonal of the covariance matrix.
		 * Return a d x n matrix where d is the dimension of the mean.
		*/
		public static Matrix SampleDiagonalVectorGaussian(
			double[] mean, double[] diag, int n){

			if(mean.Length != diag.Length){
				throw new ArgumentException("mean and diag must have the same length");
			}
			int d = mean.Length;
			Matrix m = new Matrix(d,n);
			for(int i = 0; i < d; i++){
				double meani = mean[i];
				double stdi = Math.Sqrt(diag[i]);
				for(int j = 0; j < n; j++){
					m[i, j] = Rand.Normal(meani, stdi); 
				}
			}
			return m;
		}


		// Draw n samples from U[lower, upper)
		public static double[] UniformVector(double lower, double upper, int n){
			double[] vec = new double[n];
			for(int i = 0; i < n; i++){
				// in [0, 1)
				double unit = Rand.Double();
				vec[i] = unit * (upper - lower) + lower;
			}
			return vec;
		}

		public static double Median(double[] nums){
			// O(nlog(n) ) cost. Can be O(n). Improve later if needed.
			if(nums == null || nums.Length == 0){
				throw new ArgumentException("Cannot compute a median on an empty array");
			} else if(nums.Length == 1){
				return nums[0];
			}
			int n = nums.Length;
			Array.Sort(nums);
			if(n % 2 == 0){
				// even length 
				double a = nums[n / 2 - 1];
				double b = nums[n / 2];
				return (a + b) / 2.0;
			} else{
				return nums[n / 2];
			}

		}

		/**Return an mxn matrix whose elements follow the standard normal.*/
		public static Matrix Randn(int m, int n){
			var q = Enumerable.Range(0, m * n).Select(i => Rand.Normal());
			Matrix mat = new Matrix(m,n);
			mat.SetTo(q.ToArray());
			return mat;
		}

		public static Matrix StackColumns(List<double[]> cols){
			if(cols.Count == 0){
				return new Matrix(0, 0);
			}
			// TODO: not the most efficient way...
			Vector[] vecCols = cols.Select(col => Vector.FromArray(col)).ToArray();
			Matrix m = StackColumns(vecCols);
			return m;
		}

		public static Matrix StackColumns(Vector[] cols){
			// assume all vectors have the same length 
			if(cols.Length == 0){
				return new Matrix(0, 0);
			}
			int d = cols[0].Count;
			int n = cols.Length;
			Vector stack = ConcatAll(cols);
			Matrix tran = new Matrix(n,d,stack.ToArray());
			return tran.Transpose();
		}

		public static void TestStackColumns(){
			Vector v1 = Vector.FromArray(new double[]{ 1, 2, 3 });
			Vector v2 = Vector.FromArray(new double[]{ 4, 5, 6 });
			Vector[] cols = { v1, v2 };	
			Console.WriteLine(StackColumns(cols));
		}

		public static double Product(double[] vec){
			if(vec == null || vec.Length == 0){
				throw new ArgumentException("vec must not be null or empty.");
			}
			double prod = 1;
			foreach(double v in vec){
				prod *= v;
			}
			return prod;
		}

		public static double Determinant(Matrix m){
			if(m.Rows != m.Cols){
				throw new ArgumentException("Matrix must be square");
			}

			// Infer.NET's Determinant() method of Matrix class has a bug.
			// It always returns 0.
			MNMatrix mnMat = ToMathNetMatrix(m);
			double det = mnMat.Determinant();
			return det;
		}

	}

	public static class StringUtils{
		public static string ArrayToString<T>(T[] arr){
			string s = "[";
			for(int i=0; i<arr.Length; i++){
				s += string.Format("{0}", arr[i]);
				if(i < arr.Length-1){
					s += ", ";
				}
			}
			s += "]";
			return s;
		}

		public static void PrintArray(double[] a){
			Console.Write("[");
			for(int i = 0; i < a.Length; i++){
				Console.Write(a[i]);
				if(i < a.Length - 1){
					Console.Write(", ");
				}
			}
			Console.WriteLine("]");
		}

		public static void PrintArray(bool[] a){
			Console.Write("[");
			for(int i = 0; i < a.Length; i++){
				Console.Write(a[i] ? "T" : "F");
				if(i < a.Length - 1){
					Console.Write(", ");
				}
			}
			Console.WriteLine("]");
		}

		public static void PrintVector(Vector v){
			PrintArray(v.ToArray());
		}

	}
	// end class

	// A Matlab struct loaded from MatlabReader is represented as
	// Dictionary<string, Object>
	public class MatlabStruct{
		private readonly Dictionary<string, Object> dict;

		// dict loaded from MatlabReader.Read(.)
		public MatlabStruct(Dictionary<string, object> dict){
			this.dict = dict;
		}

		private void ValidateKey(string key){
			if(!ContainsKey(key)){
				string msg = System.String.Format("key {0} does not exist", key);
				throw new ArgumentException(msg);
			}
		}

		public MatlabStruct GetStruct(string key){
			// struct is loaded as Dictionary<string, object>
			ValidateKey(key);
			Object value = dict[key];
			if(value is Dictionary<string, object>){
				// a struct 
				return new MatlabStruct(value as Dictionary<string, object>);
			}
			string msg1 = System.String.Format("key {0} is not a struct", key);
			throw new ArgumentException(msg1);
		}

		// This works on a 1d list as well.
		public Matrix GetMatrix(string key){
			// matrix [...] in Matlab is loaded as a Matrix in Infer.NET
			ValidateKey(key);
			Object value = dict[key];
			if(value is Matrix){
				return value as Matrix;
			}
			string msg1 = System.String.Format("key {0} is not a matrix", key);
			throw new ArgumentException(msg1);
		}

		public double GetDouble(string key){
			// A double is also loaded as a Matrix
			ValidateKey(key);
			Matrix mat = GetMatrix(key);
			if(mat.Cols != 1 || mat.Rows != 1){
				string msg = System.String.Format("key {0} is a matrix that cannot be cast as a scalar", key);
				throw new ArgumentException(msg);
			}
			return mat[0, 0];
		}

		public int GetInt(string key){
			double d = GetDouble(key);
			if(Math.Abs(Math.Floor(d) - d) < 1e-8){
				return (int)d;
			}
			string msg = System.String.Format("key {0} gives {1} which is not an int.",
				             key, d);
			throw new ArgumentException(msg);

		}

		public double[] Get1DDoubleArray(string key){
			// list [...] in Matlab is loaded as a Matrix. 
			ValidateKey(key);
			Matrix mat = GetMatrix(key);
			if(mat.Cols > 1 && mat.Rows > 1){
				string msg = System.String.Format("key {0} is a matrix that cannot be casted as a vector", key);
				throw new ArgumentException(msg);
			}
			if(mat.Cols == 1){
				return mat.GetColumn(0);
			} else{
//				mat.Rows == 1
				return mat.GetRow(0);
			}
		}

		public Vector Get1DVector(string key){
			double[] arr = Get1DDoubleArray(key);
			return Vector.FromArray(arr);
		}

		public string GetString(string key){
			ValidateKey(key);
			Object value = dict[key];
			if(value is String){
				return value as String;
			}
			string msg1 = System.String.Format("key {0} is not a string", key);
			throw new ArgumentException(msg1);
		}


		public MatlabStruct[,] GetStructCells(string key){
			object[,] cells = GetCells(key);
			int rows = cells.GetLength(0);
			int cols = cells.GetLength(1);
			MatlabStruct[,] structs = new MatlabStruct[rows, cols];
			for(int i = 0; i < rows; i++){
				for(int j = 0; j < cols; j++){
					var d = (Dictionary<string, object>)cells[i, j];
					structs[i, j] = new MatlabStruct(d);
				}
			}
			return structs;
		}

		public Object[,] GetCells(string key){
			ValidateKey(key);
			Object value = dict[key];
			if(value is Object[,]){
				return value as Object[,];
			} else{
				string msg1 = System.String.Format("key {0} is not a cell array (matrix)", key);
				throw new ArgumentException(msg1);
			}

		}


		public bool ContainsKey(string key){
			return dict.ContainsKey(key);
		}
		
	}
}

