using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;

namespace MicrosoftResearch.Infer.Tutorials
{
	[Example("Tutorials", "Efficient implementation of a truncated Gaussian which uses a variable for the truncation threshold.", Prefix = "2b.")]
	public class TruncatedGaussianEfficient
	{

		public void Run()
		{
			Variable<double> threshold = Variable.New<double>().Named("threshold");
			Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
			Variable.ConstrainTrue(x > threshold);
			InferenceEngine engine = new InferenceEngine();
			if (engine.Algorithm is ExpectationPropagation)
			{
				for (double thresh = 0; thresh <= 1; thresh += 0.1)
				{
					threshold.ObservedValue = thresh;
					Console.WriteLine("Dist over x given thresh of " + thresh + "=" + engine.Infer(x));
				}
			}
			else
				Console.WriteLine("This example only runs with Expectation Propagation");
		}
	}
}
