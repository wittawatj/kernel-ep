using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;

namespace MicrosoftResearch.Infer.Tutorials
{
	[Example("Tutorials","A simple first example showing the basics of Infer.NET.",Prefix="1.")]
	public class FirstExample
	{
		public void Run()
		{
			Variable<bool> firstCoin = Variable.Bernoulli(0.5).Named("firstCoin");
			Variable<bool> secondCoin = Variable.Bernoulli(0.5).Named("secondCoin");
			Variable<bool> bothHeads  = (firstCoin & secondCoin).Named("bothHeads");
			InferenceEngine ie = new InferenceEngine();
			if (!(ie.Algorithm is VariationalMessagePassing))
			{
				Console.WriteLine("Probability both coins are heads: "+ie.Infer(bothHeads));
				bothHeads.ObservedValue=false;
				Console.WriteLine("Probability distribution over firstCoin: " + ie.Infer(firstCoin));
			}
			else
				Console.WriteLine("This example does not run with Variational Message Passing");
		}
	}
}
