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
namespace KernelEP.Op{

	// T is the distribution target to map to
	public abstract class OnlineVectorMapper : UAwareVectorMapper{

		// true if the operator is uncertain on the input tuple.
		// This suggests that UpdateOperator(.) should be called.
		public abstract bool IsUncertain(params IKEPDist[] msgs);


		// Update the operator online with a new input msgs and output target
		// The target is typically obtained from an oracle.
		public abstract void UpdateVectorMapper(Vector target, params IKEPDist[] msgs);

		// The threshold used by the map for deciding its uncertainty.
		// Return null if the map is not IsUncertaintyThresholdUsed().
		public abstract double[] GetUncertaintyThreshold();

		public abstract void SetUncertaintyThreshold(params double[] thresh);

		// True if the map is threshold-based.
		public abstract bool IsThresholdBased();
	}


	// Online stacking of Bayesian linear regression functions.
	// composite pattern calling UAwareStackVectorMapper.
	public class OnlineStackBayesLinReg : BayesLinRegFM{
		private readonly BayesLinRegFM[] onlineBayes;
		private readonly UAwareStackVectorMapper stackMapper;

		public OnlineStackBayesLinReg(BayesLinRegFM[] onlineBayes){
			this.onlineBayes = onlineBayes;
			stackMapper = new UAwareStackVectorMapper(onlineBayes);
		}

		public override Vector MapToVector(params IKEPDist[] msgs){
			return stackMapper.MapToVector(msgs);
		}

		public override int GetOutputDimension(){
			return stackMapper.GetOutputDimension();
		}

		public override double[] EstimateUncertainty(params IKEPDist[] dists){
			return stackMapper.EstimateUncertainty(dists);

		}

		public override void MapAndEstimateU(out Vector mapped, 
		                                     out double[] uncertainty, params IKEPDist[] dists){
			stackMapper.MapAndEstimateU(out mapped, out uncertainty, dists);
		}

		public override bool IsUncertain(params IKEPDist[] msgs){
			// Return true if at least one of the mappers is uncertain
			var q = onlineBayes.Select(map => map.IsUncertain());
			bool[] uncertains = q.ToArray();
			return MatrixUtils.Or(uncertains);
		}

		public override void UpdateVectorMapper(Vector target, params IKEPDist[] msgs){
			// update each internal mapper 
			foreach(OnlineVectorMapper map in onlineBayes){
				map.UpdateVectorMapper(target, msgs);
			}
		}

		public override double[] GetUncertaintyThreshold(){
			// Get the threshold for only the first one of each mapper 
			var q = onlineBayes.Select(map => map.GetUncertaintyThreshold()[0]);
			double[] thresholds = q.ToArray();
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

		public override bool IsThresholdBased(){
			// true if all mappers are threshold based 
			var q = onlineBayes.Select(map => map.IsThresholdBased());
			bool[] tb = q.ToArray();
			return MatrixUtils.And(tb);
		}

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
		public abstract bool IsUncertain(params IKEPDist[] msgs);

		// Update the operator online with a new input msgs and output target
		// The target is typically obtained from an oracle.
		public abstract void UpdateOperator(T target, params IKEPDist[] msgs);
	}

	public class PrimalGPOnlineMapper<T> : OnlineDistMapper<T>
		where T:IKEPDist{

		private readonly BayesLinRegFM bayesSuffMapper;

		protected PrimalGPOnlineMapper(BayesLinRegFM bayesVecMap, 
		                               DistBuilder<T> distBuilder)
			: base(bayesVecMap, distBuilder){
			this.bayesSuffMapper = bayesVecMap;
		}


		public override T MapToDist(params IKEPDist[] msgs){
			// stack of predicted regression targets (expected sufficient stats)
			Vector suffPredicts = bayesSuffMapper.MapToVector(msgs);
			// mapped vector and distBuilder must be compatible.
			T outDist = distBuilder.FromStat(suffPredicts);
			return outDist;
		}

		public override double[] EstimateUncertainty(params IKEPDist[] msgs){
			return bayesSuffMapper.EstimateUncertainty(msgs);
		}

		public override bool IsUncertain(params IKEPDist[] msgs){
			return bayesSuffMapper.IsUncertain(msgs);
		}

		public override void UpdateOperator(T target, params IKEPDist[] msgs){
			throw new NotImplementedException();
//			bayesSuffMapper.UpdateVectorMapper( )
		}

	}



}

