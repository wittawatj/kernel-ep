// (C) Copyright 2011 Microsoft Research Cambridge
using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using CyclingModels;

namespace CyclingModels
{
	public class CyclistWithEvidence : CyclistTraining
	{
		protected Variable<bool> Evidence;

		public override void CreateModel()
		{
			Evidence = Variable.Bernoulli(0.5);
			using (Variable.If(Evidence))
			{
				base.CreateModel();
			}
		}

		public double InferEvidence(double[] trainingData)
		{
			double logEvidence;
			ModelData posteriors = base.InferModelData(trainingData);
			logEvidence = InferenceEngine.Infer<Bernoulli>(Evidence).LogOdds;

			return logEvidence;
		}
	}

	public class CyclistMixedWithEvidence : CyclistMixedTraining
	{
		protected Variable<bool> Evidence;

		public override void CreateModel()
		{
			Evidence = Variable.Bernoulli(0.5);
			using (Variable.If(Evidence))
			{
				base.CreateModel();
			}
		}

		public double InferEvidence(double[] trainingData)
		{
			double logEvidence;
			ModelDataMixed posteriors = base.InferModelData(trainingData);
			logEvidence = InferenceEngine.Infer<Bernoulli>(Evidence).LogOdds;

			return logEvidence;
		}
	}
}
