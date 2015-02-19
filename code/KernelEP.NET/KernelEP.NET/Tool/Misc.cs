using System;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;

namespace KernelEP{
	public static class MatrixUtils{

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
			Matrix big = new Matrix(totalRows, totalCols);
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

	}
	// interface marking that the class's objects have a summary
	public interface IHasShortSummary{
		string ShortSummary();
	}

	public static class PrintUtils{
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

