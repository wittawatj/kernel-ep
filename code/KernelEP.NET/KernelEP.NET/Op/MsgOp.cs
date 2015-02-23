using System;
using System.Collections.Generic;
using System.Linq;
using KernelEP.Tool;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

namespace KernelEP.Op{
	
	// class to control all operators in our framework
	public static class OpControl{
		// a map from a message operator type to its operator parameters
		private readonly static Dictionary<Type, MsgOpInstance> opInternals
		= new Dictionary<Type, MsgOpInstance>();

		static OpControl(){
			// default operator internals
//			opInternals.Add(KEPLogisticOp, some_thing )
		}

		public static void Add(Type t, MsgOpInstance oi){
			if(oi == null){
				throw new ArgumentException("Operator internal cannot be null.");
			}
			opInternals.Add(t, oi);
		}
		public static void Set(Type t, MsgOpInstance oi){
			if(oi == null){
				throw new ArgumentException("Operator internal cannot be null.");
			}
			opInternals[t] = oi;
		}

		public static MsgOpInstance Get(Type t){
			if(!opInternals.ContainsKey(t)){
				throw new ArgumentException("Parameters undefined for messages operator: " + t);
			}
			return opInternals[t];
		}
	}


	// Equivalent class in Matlab code is FactorOperator.
	public abstract class OpParamsBase{

	}


	// Internal components (e.g., DistMapper to each variable) of a message
	// passing operator (e.g., XXXOp). This is useful in setting DistMapper for
	// each target variable to send to. One object of this class wraps all
	// DistMapper's.
	// A, B are types of the distributions.
	public class OpParams<A, B> : OpParamsBase 
		where A : IKEPDist where B : IKEPDist{
		protected DistMapper<A> DistMapper0;
		protected DistMapper<B> DistMapper1;

		public OpParams(DistMapper<A> dm0, DistMapper<B> dm1){
			this.DistMapper0 = dm0;
			this.DistMapper1 = dm1;

		}

		public DistMapper<A> GetDistMapper0(){
			return this.DistMapper0;
		}

		public DistMapper<B> GetDistMapper1(){
			return this.DistMapper1;
		}
	}

	// A message operator instance.
	// This is intended to be wrapped by a static class implementing an Infer.NET's 
	// message operator. It is easier to manage objects rather than static classes.
	public abstract class MsgOpInstance{

	}


}

