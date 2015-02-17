/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.StringOfLength(int, DiscreteChar)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "StringOfLength")]
    [Quality(QualityBand.Experimental)]
    public static class StringOfLengthOp
    {
        #region EP messages

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>str</c> conditioned on the given values.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, int length)
        {
            Argument.CheckIfNotNull(allowedChars, "allowedChars");
            Argument.CheckIfValid(allowedChars.IsPartialUniform(), "allowedChars", "The set of allowed characters must be passed as a partial uniform distribution.");
            
            return StringDistribution.Repeat(allowedChars, length, length);
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="length">Incoming message from <c>length</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(length) p(length) factor(str,length,allowedChars)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(DiscreteChar allowedChars, Discrete length)
        {
            Argument.CheckIfNotNull(allowedChars, "allowedChars");
            Argument.CheckIfNotNull(length, "length");
            Argument.CheckIfValid(allowedChars.IsPartialUniform(), "allowedChars", "The set of allowed characters must be passed as a partial uniform distribution.");

            double logNormalizer = allowedChars.GetLogAverageOf(allowedChars);
            var oneCharacter = StringAutomaton.ConstantOnElementLog(logNormalizer, allowedChars);
            var manyCharacters = StringAutomaton.Repeat(oneCharacter, length.GetWorkspace());
            return StringDistribution.FromWorkspace(manyCharacters);
        }

        /// <summary>EP message to <c>length</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>length</c> as the random arguments are varied. The formula is <c>proj[p(length) sum_(str) p(str) factor(str,length,allowedChars)]/p(length)</c>.</para>
        /// </remarks>
        public static Discrete LengthAverageConditional(StringDistribution str, DiscreteChar allowedChars, Discrete result)
        {
            Vector resultProbabilities = result.GetWorkspace();
            for (int length = 0; length < result.Dimension; ++length)
            {
                StringDistribution factor = StringDistribution.Repeat(allowedChars, length, length);
                resultProbabilities[length] = Math.Exp(factor.GetLogAverageOf(str));
            }
            
            result.SetProbs(resultProbabilities);
            return result;
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="length">Incoming message from <c>length</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(length,str) p(length,str) factor(str,length,allowedChars) / sum_str p(str) messageTo(str))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(DiscreteChar allowedChars, Discrete length, StringDistribution str)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <param name="str">Constant value for <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(str,length,allowedChars))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(DiscreteChar allowedChars, int length, string str)
        {
            StringDistribution toStr = StrAverageConditional(allowedChars, length);
            return toStr.GetLogProb(str);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="allowedChars">Constant value for <c>allowedChars</c>.</param>
        /// <param name="length">Incoming message from <c>length</c>.</param>
        /// <param name="str">Constant value for <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(length) p(length) factor(str,length,allowedChars))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(DiscreteChar allowedChars, Discrete length, string str)
        {
            StringDistribution toStr = StrAverageConditional(allowedChars, length);
            return toStr.GetLogProb(str);
        }

        #endregion
    }
}
