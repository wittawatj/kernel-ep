using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;

namespace MicrosoftResearch.Infer.Tutorials
{
	[Example("Tutorials", "Model comparison example to determine if a new medical treatment is effective.", Prefix = "5.")]
	public class ClinicalTrial
	{
		public void Run()
		{
			// Data from clinical trial
			VariableArray<bool> controlGroup = 
				Variable.Observed(new bool[] { false, false, true, false, false }).Named("controlGroup");
			VariableArray<bool> treatedGroup = 
				Variable.Observed(new bool[] { true, false, true, true, true }).Named("treatedGroup");
			Range i = controlGroup.Range.Named("i"); Range j = treatedGroup.Range.Named("j");

			// Prior on being effective treatment
			Variable<bool> isEffective = Variable.Bernoulli(0.5).Named("isEffective");
			Variable<double> probIfTreated, probIfControl;
			using (Variable.If(isEffective))
			{
				// Model if treatment is effective
				probIfControl = Variable.Beta(1, 1).Named("probIfControl");
				controlGroup[i] = Variable.Bernoulli(probIfControl).ForEach(i);
				probIfTreated = Variable.Beta(1, 1).Named("probIfTreated");
				treatedGroup[j] = Variable.Bernoulli(probIfTreated).ForEach(j);
			}
			using (Variable.IfNot(isEffective))
			{
				// Model if treatment is not effective
				Variable<double> probAll = Variable.Beta(1, 1).Named("probAll");
				controlGroup[i] = Variable.Bernoulli(probAll).ForEach(i);
				treatedGroup[j] = Variable.Bernoulli(probAll).ForEach(j);
			}
			InferenceEngine ie = new InferenceEngine();
			if (!(ie.Algorithm is GibbsSampling))
			{
				Console.WriteLine("Probability treatment has an effect = " + ie.Infer(isEffective));
				Console.WriteLine("Probability of good outcome if given treatment = " 
										+ (float)ie.Infer<Beta>(probIfTreated).GetMean());
				Console.WriteLine("Probability of good outcome if control = " 
										+ (float)ie.Infer<Beta>(probIfControl).GetMean());
			}
			else
				Console.WriteLine("This model is not supported by Gibbs sampling.");
		}
	}
}
