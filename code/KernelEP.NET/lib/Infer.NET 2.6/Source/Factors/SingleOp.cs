namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Diagnostics;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.Single(String)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Single")]
    [Quality(QualityBand.Experimental)]
    public static class SingleOp
    {
        /// <summary>EP message to <c>character</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>The outgoing EP message to the <c>character</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>character</c> as the random arguments are varied. The formula is <c>proj[p(character) sum_(str) p(str) factor(character,str)]/p(character)</c>.</para>
        /// </remarks>
        public static DiscreteChar CharacterAverageConditional(StringDistribution str)
        {
            Argument.CheckIfNotNull(str, "str");

            Vector resultlogProb = PiecewiseVector.Constant(char.MaxValue + 1, double.NegativeInfinity);
            StringAutomaton probFunc = str.GetProbabilityFunction();
            StringAutomaton.EpsilonClosure startEpsilonClosure = probFunc.Start.GetEpsilonClosure();
            for (int stateIndex = 0; stateIndex < startEpsilonClosure.Size; ++stateIndex)
            {
                StringAutomaton.State state = startEpsilonClosure.GetStateByIndex(stateIndex);
                double stateLogWeight = startEpsilonClosure.GetStateLogWeightByIndex(stateIndex);
                for (int transitionIndex = 0; transitionIndex < state.Transitions.Count; ++transitionIndex)
                {
                    StringAutomaton.Transition transition = state.Transitions[transitionIndex];
                    if (!transition.IsEpsilon)
                    {
                        StringAutomaton.State destState = probFunc.States[transition.DestinationStateIndex];
                        StringAutomaton.EpsilonClosure destStateClosure = destState.GetEpsilonClosure();
                        if (!double.IsNegativeInfinity(destStateClosure.EndLogWeight))
                        {
                            double logWeight = stateLogWeight + transition.LogWeight + destStateClosure.EndLogWeight;
                            resultlogProb = LogSumExp(resultlogProb, transition.ElementDistribution.GetInternalDiscrete().GetLogProbs(), logWeight);
                        }
                    }                        
                }
            }

            if (resultlogProb.All(double.IsNegativeInfinity))
            {
                throw new AllZeroException("An input distribution assigns zero probability to all single character strings.");
            }

            Vector resultProb = PiecewiseVector.Zero(char.MaxValue + 1);
            resultProb.SetToFunction(resultlogProb, Math.Exp);
            return DiscreteChar.FromVector(resultProb);
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="character">Incoming message from <c>character</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(character) p(character) factor(character,str)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(DiscreteChar character)
        {
            return StringDistribution.Char(character);
        }

        private static Vector LogSumExp(Vector logValues1, Vector logValues2, double values2LogScale)
        {
            Debug.Assert(logValues1.Count == logValues2.Count);
            logValues1.SetToFunction(logValues1, logValues2, (x, y) => MMath.LogSumExp(x, values2LogScale + y));
            return logValues1;
        }
    }
}
