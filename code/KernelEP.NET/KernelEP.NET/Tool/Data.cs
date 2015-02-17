using System;
using System.Diagnostics;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;


// This file contains data structure for representing various types of data,
// mainly messages.

namespace KernelEP.Tool{

	// T parametrises the type of each data point
	public abstract class Instances<T> {
		public static string MATLAB_CLASS = "Instances";
		// Return data instances specified by the indices in Ind. Ind is a
		// list of indices. The type of Data is not restricted (depending on
		// the implmentation).
		public abstract List<T> Get(params int[] ind);

		// Return all data instances.
		public abstract List<T> GetAll();

		// Return data instances specified by the indices in Ind in the form
		// of Instances. Ind is a list of indices.
		public abstract Instances<T> GetInstances(params int[] ind);

		// total number of instances
		public abstract int Count();

		public int Length(){
			return Count();
		}

//		public static  Instances<T> FromMatlabStruct(MatlabStruct s){
//
//			string className = s.GetString("className");
//			if(className.Equals("DistArray")){
//				throw new ArgumentException("what to do here ?");
//			}else{
//				throw new ArgumentException("Unknown class for Instances.");
//			}
//
//		}
	}

	// array of distributions
	// Has a corresponding Matlab class
	public class DistArray<T> : Instances<T>
		where T : IKEPDist{

		private List<T> dists;

		public DistArray(List<T> dists){
			this.dists = dists;
		}

		public List<T> GetDists(){
			return dists;
		}

		public override List<T> Get(params int[] ind){
			var qresult = 
				from i in ind 
				select dists[i];
			return qresult.ToList();
		}

		public override List<T> GetAll(){
			return dists;
		}

		public override Instances<T> GetInstances(params int[] ind){
			return new DistArray<T>(this.Get(ind));
		}

		public override int Count(){
			return dists.Count;
		}

		public new static  DistArray<T> FromMatlabStruct(MatlabStruct s){
//			s = struct();
//			s.className=class(this);
//			distCell = cell(1, length(this.distArray));
//			for i=1:length(this.distArray)
//					dist = this.distArray(i);
//					distCell{i} = dist.toStruct();
//			end
//			s.distArray = distCell;
//			s.mean = this.mean;
//			s.variance = this.variance;
//
			string className = s.GetString("className");
			if(!className.Equals("DistArray")){
				throw new ArgumentException("The input does not represent a " + "DistArray");
			}
			object[,] distCell =s.GetCells("distArray");
			List<T> dists = new List<T>();
			for(int i=0; i<distCell.Length; i++){
				var distDict = (Dictionary<string, object>)distCell[0, i];
				MatlabStruct distStruct = new MatlabStruct(distDict);
				IKEPDist disti = KEPDist.FromMatlabStruct(distStruct);
				dists.Add((T)disti);
			}
			return new DistArray<T>(dists);
		}


	}

	// Has a corresponding Matlab class
	public class TensorInstances<T1, T2> : Instances<Tuple<T1, T2>>
		where T1 : IKEPDist
		where T2 : IKEPDist{

		private List<T1> distArray1;
		private List<T2> distArray2;

		public static string MATLAB_CLASS = "TensorInstances";


		public TensorInstances(List<T1> d1, List<T2> d2){
			if(d1.Count != d2.Count){

				throw new ArgumentException("d1 and d2 do not have the same length.");
			}
			this.distArray1 = d1;
			this.distArray2 = d2;
		}


		public override List<Tuple<T1, T2>> Get(params int[] ind){
			var qresult =
				from i in ind 
				select new Tuple<T1, T2>(distArray1[i], distArray2[i]);
			return qresult.ToList();
		}

		public override List<Tuple<T1, T2>> GetAll(){
			var qresult =
				from i in Enumerable.Range(0, distArray1.Count)
				select new Tuple<T1, T2>(distArray1[i], distArray2[i]);
			return qresult.ToList();
		}

		public override Instances<Tuple<T1, T2>> GetInstances(params int[] ind){
			var q1 = from i in ind select distArray1[i];
			var q2 = from i in ind select distArray2[i];
			List<T1> l1 = q1.ToList();
			List<T2> l2 = q2.ToList();
			return new TensorInstances<T1, T2>(l1, l2);

		}

		public override int Count(){
			Debug.Assert(distArray1.Count == distArray2.Count);
			return distArray1.Count;
		}

		public static  TensorInstances<T1, T2> FromMatlabStruct(MatlabStruct s){
//			s = struct();
//			s.className=class(this);
//			instancesCount = length(this.instancesCell);
//			cellStruct = cell(1, instancesCount);
//			for i=1:instancesCount
//					cellStruct = this.instancesCell{i}.toStruct();
//			end
//			s.instancesCell = cellStruct;

			string className = s.GetString("className");
			if(!className.Equals(MATLAB_CLASS)){
				throw new ArgumentException("The input does not represent a " +
				typeof(TensorInstances<T1, T2>));
			}
			int instancesCount = s.GetInt("instancesCount");
			if(instancesCount != 2){
				throw new ArgumentException("expect instancesCount to be 2.");
			}
			object[,] instancesCell = s.GetCells("instancesCell");
			if(instancesCell.Length != 2){
				throw new ArgumentException("instancesCell does not have length 2.");
			}
			var da1Dict = (Dictionary<string, object>) instancesCell[0, 0];
			var da2Dict = (Dictionary<string, object>) instancesCell[0, 1];

			// assume instancesCell contains DistArray's
			DistArray<T1> da1 = DistArray<T1>.FromMatlabStruct(
				new MatlabStruct(da1Dict));
			DistArray<T2> da2 = DistArray<T2>.FromMatlabStruct(
				new MatlabStruct(da2Dict));
			return new TensorInstances<T1, T2>(da1.GetDists(), da2.GetDists());
		}

	}

	// A class representing a tuple of incoming messages
	public class IncomingMsgs{
		private readonly IKEPDist[] msgs;

		public IncomingMsgs(params IKEPDist[] msgs){
			this.msgs = msgs;
		}

		public IKEPDist[] GetMessages(){
			return msgs;
		}

		public int NumMsgs(){
			// Return the number of incoming messages in this tuple
			return msgs.Length;
		}
	}
}

