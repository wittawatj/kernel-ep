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
		private OnlineVectorMapper[] onlineVecMappers;

		protected PrimalGPOnlineMapper(DistBuilder<T> distBuilder,  
			params OnlineVectorMapper[] onlineVecMaps)
			: base(null, distBuilder){

			this.onlineVecMappers = onlineVecMaps;
		}


		public override T MapToDist(params IKEPDist[] msgs){
			throw new NotImplementedException();
		}

		public override double[] EstimateUncertainty(params IKEPDist[] msgs){
			throw new System.NotImplementedException();
		}

		public override bool IsUncertain(params IKEPDist[] msgs){
			throw new System.NotImplementedException();
		}

		public override void UpdateOperator(T target, params IKEPDist[] msgs){
			throw new System.NotImplementedException();
		}

	}



}

