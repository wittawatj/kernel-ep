using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;

namespace MicrosoftResearch.Infer.Tutorials
{
	[Example("Tutorials", "Inefficient implementation of a truncated Gaussian distribution.",Prefix="2a.")]
	public class TruncatedGaussian
	{
		public void Run()
		{
			for (double thresh = 0; thresh <= 1; thresh += 0.1)
			{
				Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
				Variable.ConstrainTrue(x > thresh);
				InferenceEngine engine = new InferenceEngine();
				if (engine.Algorithm is ExpectationPropagation)
					Console.WriteLine("Dist over x given thresh of " + thresh + "=" + engine.Infer(x));
				else
					Console.WriteLine("This example only runs with Expectation Propagation");
			}
		}

	}


}
