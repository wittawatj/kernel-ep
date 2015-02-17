// (C) Copyright 2011 Microsoft Research Cambridge
using System;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace CyclingModels
{
	public class TwoCyclistsTraining
	{
		private CyclistTraining cyclist1, cyclist2;

		public void CreateModel()
		{
			cyclist1 = new CyclistTraining();
			cyclist1.CreateModel();
			cyclist2 = new CyclistTraining();
			cyclist2.CreateModel();
		}

		public void SetModelData(ModelData modelData)
		{
			cyclist1.SetModelData(modelData);
			cyclist2.SetModelData(modelData);
		}

		public ModelData[] InferModelData(double[] trainingData1,
										  double[] trainingData2)
		{
			ModelData[] posteriors = new ModelData[2];

			posteriors[0] = cyclist1.InferModelData(trainingData1);
			posteriors[1] = cyclist2.InferModelData(trainingData2);

			return posteriors;
		}

	}

	public class TwoCyclistsPrediction
	{
		private CyclistPrediction cyclist1, cyclist2;
		private Variable<double> TimeDifference;
		private Variable<bool> Cyclist1IsFaster;
		private InferenceEngine CommonEngine;

		public void CreateModel()
		{
			CommonEngine = new InferenceEngine();

			cyclist1 = new CyclistPrediction() { InferenceEngine = CommonEngine };
			cyclist1.CreateModel();
			cyclist2 = new CyclistPrediction() { InferenceEngine = CommonEngine };
			cyclist2.CreateModel();

			TimeDifference = cyclist1.TomorrowsTime - cyclist2.TomorrowsTime;
			Cyclist1IsFaster = cyclist1.TomorrowsTime < cyclist2.TomorrowsTime;
		}

		public void SetModelData(ModelData[] modelData)
		{
			cyclist1.SetModelData(modelData[0]);
			cyclist2.SetModelData(modelData[1]);
		}

		public Gaussian[] InferTomorrowsTime()
		{
			Gaussian[] tomorrowsTime = new Gaussian[2];

			tomorrowsTime[0] = cyclist1.InferTomorrowsTime();
			tomorrowsTime[1] = cyclist2.InferTomorrowsTime();
			return tomorrowsTime;
		}

		public Gaussian InferTimeDifference()
		{
			return CommonEngine.Infer<Gaussian>(TimeDifference);
		}

		public Bernoulli InferCyclist1IsFaster()
		{
			return CommonEngine.Infer<Bernoulli>(Cyclist1IsFaster);
		}
	}
}
