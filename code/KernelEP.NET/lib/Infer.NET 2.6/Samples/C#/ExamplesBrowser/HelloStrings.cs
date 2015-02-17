namespace MicrosoftResearch.Infer.Tutorials
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;

    [Example("String tutorials", "An introduction to inference over string variables", Prefix = "1.")]
    public class HelloStrings
    {
        public void Run()
        {
            Variable<string> str1 = Variable.StringUniform().Named("str1");
            Variable<string> str2 = Variable.StringUniform().Named("str2");
            Variable<string> text = (str1 + " " + str2).Named("text");
            text.ObservedValue = "Hello uncertain world";

            var engine = new InferenceEngine();
            engine.Compiler.RecommendedQuality = QualityBand.Experimental;

            if (engine.Algorithm is ExpectationPropagation)
            {
                Console.WriteLine("str1: {0}", engine.Infer(str1));
                Console.WriteLine("str2: {0}", engine.Infer(str2));

                var distOfStr1 = engine.Infer<StringDistribution>(str1);
                foreach (var s in new[] { "Hello", "Hello uncertain", "Hello uncertain world" })
                {
                    Console.WriteLine("P(str1 = '{0}') = {1}", s, distOfStr1.Evaluate(s));
                }
            }
            else
            {
                Console.WriteLine("This example only runs with Expectation Propagation");
            }
        }
    }
}
