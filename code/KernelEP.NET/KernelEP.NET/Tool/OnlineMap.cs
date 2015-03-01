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

// This file contains classes related to online-active message operators
namespace KernelEP.Tool{

	// T is the distribution target to map to
	public abstract class OnlineVectorMapper : UAwareVectorMapper{

		// true if the operator is uncertain on the input tuple.
		// This suggests that UpdateOperator(.) should be called.
//		public abstract bool IsUncertain(params IKEPDist[] msgs);


		// Update the operator online with a new input msgs and output target
		// The target is typically obtained from an oracle.
		public abstract void UpdateVectorMapper(Vector target, params IKEPDist[] msgs);

		// The threshold used by the map for deciding its uncertainty.
		// Return null if the map is not IsUncertaintyThresholdUsed().
		public abstract double[] GetUncertaintyThreshold();

		public abstract void SetUncertaintyThreshold(params double[] thresh);

		// True if the map is threshold-based.
//		public abstract bool IsThresholdBased();

		/** 
		 * Return false if the mapper is not ready for an online learning.
		 * This will be the case in the beginning for an operator requiring an 
		 * initial minibatch training.
		*/
		public abstract bool IsOnlineReady();
	}


	// Online stacking of Bayesian linear regression functions.
	// composite pattern calling UAwareStackVectorMapper.
	public class OnlineStackBayesLinReg : OnlineVectorMapper{
		private readonly BayesLinRegFM[] onlineBayes;

		public OnlineStackBayesLinReg(params BayesLinRegFM[] onlineBayes){
			this.onlineBayes = onlineBayes;
		}
			
		public void SetFeatures(int[] numFeatures){
			for(int i=0; i<onlineBayes.Length; i++){
				onlineBayes[i].InnerFeatures = numFeatures[0];
				onlineBayes[i].OuterFeatures = numFeatures[1];
			}

		}
		public void SetMinibatchFeatures(int[] numFeatures){
			for(int i=0; i<onlineBayes.Length; i++){
				onlineBayes[i].MinibatchInnerFeatures = numFeatures[0];
				onlineBayes[i].MinibatchOuterFeatures = numFeatures[1];
			}

		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			double[] outputs = new double[onlineBayes.Length];
			for(int i=0; i<onlineBayes.Length; i++){
				outputs[i] = onlineBayes[i].MapToDouble(msgs);
			}
			return Vector.FromArray(outputs);
		}
		public  Vector MapToVector(Vector[] randomFeatures){
			double[] outputs = new double[onlineBayes.Length];
			for(int i=0; i<onlineBayes.Length; i++){
				outputs[i] = onlineBayes[i].MapToDouble(randomFeatures[i]);
			}
			return Vector.FromArray(outputs);
		}

		public override int GetOutputDimension(){
			return onlineBayes.Length;
		}

		public override double[] EstimateUncertainty(params IKEPDist[] dists){
			double[] un = new double[onlineBayes.Length];
			for(int i=0; i<onlineBayes.Length; i++){
				un[i] = onlineBayes[i].EstimateUncertainty(dists)[0];
			}
			return un;
		}

		/**Estimate uncertainty from a list of feature vectors from 
		GenAllRandomFeatures(.) */
		public double[] EstimateUncertainty(Vector[] randomFeatures){
			// randomFeatures, one vector for each output
			double[] un = new double[onlineBayes.Length];
			for(int i=0; i<onlineBayes.Length; i++){
				un[i] = onlineBayes[i].EstimateUncertainty(randomFeatures[i])[0];
			}
			return un;
		}

		public void SetOnlineBatchSizeTrigger(int size){

			for(int i=0; i<onlineBayes.Length; i++){
				onlineBayes[i].SetOnlineBatchSizeTrigger(size);
			}
		}


		/**Generate random feature vectors for all outputs. */
		public Vector[] GenAllRandomFeatures(params IKEPDist[] dists){
			var q = onlineBayes.Select(b => b.GenRandomFeatures(dists));
			Vector[] features = q.ToArray();
			return features;
		}

		public override void MapAndEstimateU(out Vector mapped, 
			out double[] uncertainty, out bool uncertain, params IKEPDist[] dists){

			Vector[] features = GenAllRandomFeatures(dists);
			uncertainty = EstimateUncertainty(features);
			uncertain = IsUncertain(features);
			mapped = MapToVector(features);
		}

//		public override bool IsUncertain(params IKEPDist[] msgs){
//			// Return true if at least one of the mappers is uncertain
//			for(int i=0; i<onlineBayes.Length; i++){
//				if(onlineBayes[i].IsUncertain(msgs)){
//					return true;
//				}
//			}
//			return false;
//		}

		private  bool IsUncertain(Vector[] features){
			// Return true if at least one of the mappers is uncertain
			for(int i=0; i<onlineBayes.Length; i++){
				BayesLinRegFM bi = onlineBayes[i];
				double[] thresh = bi.GetUncertaintyThreshold();
				if(thresh.Length != 1){
					throw new ArgumentException("Threshold from a Bayesian linear regression should have just one number.");
				}
				double t= thresh[0];
				double[] un = bi.EstimateUncertainty();
				if(un.Length != 1){
					throw new ArgumentException("Uncertainty from a Bayesian linear regression should have just one number");
				}
				double u = un[0];
				if(u >= t){
					return true;
				}
			}
			return false;
		}

		public override bool IsOnlineReady(){
			// Return false if at least one of the mappers is not ready
			foreach(BayesLinRegFM m in onlineBayes){
				if(!m.IsOnlineReady()){
					return false;
				}
			}
			return true;
		}
		public override void UpdateVectorMapper(Vector target, params IKEPDist[] msgs){
			// update each internal mapper
			if(target.Count != onlineBayes.Length){
				throw new ArgumentException("Require target length == number of nested Bayes learners");
			}
			for(int i=0; i<onlineBayes.Length; i++){
				Vector oneDTarget = Vector.FromArray( new double[]{target[i] } );
				BayesLinRegFM map = onlineBayes[i];
				map.UpdateVectorMapper(oneDTarget, msgs);
			}

		}
		public  void UpdateVectorMapper(Vector target, Vector[] randomFeatures){
			// update each internal mapper
			if(target.Count != onlineBayes.Length){
				throw new ArgumentException("Require target length == number of nested Bayes learners");
			}
			throw new NotImplementedException();

		}


		public override double[] GetUncertaintyThreshold(){
			// Get the threshold for only the first one of each mapper 
			double[] thresholds = new double[onlineBayes.Length];
			for(int i=0; i<onlineBayes.Length; i++){
				double[] t = onlineBayes[i].GetUncertaintyThreshold();
				thresholds[i] = t[0];
			}
			return thresholds;
		}

		public override void SetUncertaintyThreshold(params double[] thresh){
			// set only the first threshold of each map 
			if(thresh.Length != onlineBayes.Length){
				throw new ArgumentException("threshold length does not match number of internal mappers");
			}
			for(int i = 0; i < onlineBayes.Length; i++){
				OnlineVectorMapper map = onlineBayes[i];
				double[] t = map.GetUncertaintyThreshold();
				t[0] = thresh[i];
				map.SetUncertaintyThreshold(t);
			}
		}

//		public override bool IsThresholdBased(){
//			// true if all mappers are threshold based 
//			var q = onlineBayes.Select(map => map.IsThresholdBased());
//			bool[] tb = q.ToArray();
//			return MatrixUtils.And(tb);
//		}

	}

	// T is the distribution target to map to
	public abstract class OnlineDistMapper<T> : UAwareDistMapper<T>
		where T: IKEPDist{

		protected OnlineDistMapper(OnlineVectorMapper suffMapper, 
		                           DistBuilder<T> distBuilder)
			: base(suffMapper, distBuilder){

		}

		// true if the operator is uncertain on the input tuple.
		// This suggests that UpdateOperator(.) should be called.
//		public abstract bool IsUncertain(params IKEPDist[] msgs);

		// Update the operator online with a new input msgs and output target
		// The target is typically obtained from an oracle.
		public abstract void UpdateOperator(T target, params IKEPDist[] msgs);

		/** 
		 * Return false if the mapper is not ready for an online learning.
		 * This will be the case in the beginning for an operator requiring an 
		 * initial minibatch training.
		*/
		public abstract bool IsOnlineReady();
		public abstract double[] GetUncertaintyThreshold();

//		public abstract void SetUncertaintyThreshold(params double[] thresh);


	}

	public class PrimalGPOnlineMapper<T> : OnlineDistMapper<T>
		where T:IKEPDist{

		private readonly OnlineStackBayesLinReg bayesSuffMappers;

		public PrimalGPOnlineMapper(OnlineStackBayesLinReg bayesVecMap, 
		                               DistBuilder<T> distBuilder)
			: base(bayesVecMap, distBuilder){
			this.bayesSuffMappers = bayesVecMap;
		}

		/**Generate random feature vectors for all outputs. */
		public Vector[] GenAllRandomFeatures(params IKEPDist[] msgs){
			Vector[] features = bayesSuffMappers.GenAllRandomFeatures(msgs);
			return features;
		}
		public override T MapToDist(params IKEPDist[] msgs){
			// stack of predicted regression targets (expected sufficient stats)
			Vector suffPredicts = bayesSuffMappers.MapToVector(msgs);
			// mapped vector and distBuilder must be compatible.
			T outDist = distBuilder.FromStat(suffPredicts);
			return outDist;
		}

		public T MapToDist(Vector suff){

			T outDist = distBuilder.FromStat(suff);
			return outDist;
		}
		public T MapToDistFromRandomFeatures(Vector[] features){
			Vector suff= bayesSuffMappers.MapToVector(features);
			return MapToDist(suff);
		}

		public double[] EstimateUncertainty(Vector[] features){
			return bayesSuffMappers.EstimateUncertainty(features);
		}

		public override double[] EstimateUncertainty(params IKEPDist[] msgs){
			return bayesSuffMappers.EstimateUncertainty(msgs);
		}

		public void SetOnlineBatchSizeTrigger(int size){
			bayesSuffMappers.SetOnlineBatchSizeTrigger(size);
		}

		public void SetFeatures(int[] numFeatures){
			bayesSuffMappers.SetFeatures(numFeatures);

		}
		public void SetMinibatchFeatures(int[] numFeatures){
			bayesSuffMappers.SetMinibatchFeatures(numFeatures);

		}

//		public override bool IsUncertain(params IKEPDist[] msgs){
//			return bayesSuffMappers.IsUncertain(msgs);
//		}

//		public bool IsUncertain(Vector[] features){
//			return bayesSuffMappers.IsUncertain(features);
//		}

		public  void UpdateOperator(T target, Vector[] features){
			Vector suff = distBuilder.GetStat(target);
			throw new NotImplementedException("should implement this for efficiency");
		}

		public override void UpdateOperator(T target, params IKEPDist[] msgs){
			Vector suff = distBuilder.GetStat(target);
			bayesSuffMappers.UpdateVectorMapper(suff, msgs);
		}
		public override bool IsOnlineReady(){
			return bayesSuffMappers.IsOnlineReady();
		}

		public override double[] GetUncertaintyThreshold(){
			return bayesSuffMappers.GetUncertaintyThreshold();
		}

//		public override bool IsThresholdBased(){
//			return bayesSuffMappers.IsThresholdBased();
//		}

		public override void MapAndEstimateU(out T mapped, 
			out double[] uncertainty, out bool uncertain, params IKEPDist[] dists){

			Vector suffPredicts;
			bayesSuffMappers.MapAndEstimateU(out suffPredicts, out uncertainty, 
				out uncertain, dists);
			mapped = distBuilder.FromStat(suffPredicts);
		}
	}



}

