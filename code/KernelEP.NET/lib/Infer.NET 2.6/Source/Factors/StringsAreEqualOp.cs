/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.AreEqual(String, String)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "AreEqual", typeof(string), typeof(string))]
    [Quality(QualityBand.Experimental)]
    public static class StringsAreEqualOp
    {
        #region EP messages

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(str1,str2) p(str1,str2) factor(areEqual,str1,str2)]/p(areEqual)</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(StringDistribution str1, StringDistribution str2)
        {
            return Bernoulli.FromLogOdds(MMath.LogitFromLog(str1.GetLogAverageOf(str2)));
        }

        /// <summary>EP message to <c>str1</c>.</summary>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str1</c> as the random arguments are varied. The formula is <c>proj[p(str1) sum_(str2,areEqual) p(str2,areEqual) factor(areEqual,str1,str2)]/p(str1)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static StringDistribution Str1AverageConditional(
            StringDistribution str2, [SkipIfUniform] Bernoulli areEqual, StringDistribution result)
        {
            StringDistribution uniform = StringDistribution.Any();
            double probNotEqual = areEqual.GetProbFalse();
            if (probNotEqual > 0.5)
            {
                throw new NotImplementedException("Non-equality case is not yet supported.");
            }

            double logWeight1 = Math.Log(1 - (2 * probNotEqual));
            double logWeight2 = Math.Log(probNotEqual) + uniform.GetLogNormalizer();
            result.SetToSumLog(logWeight1, str2, logWeight2, uniform);
            return result;
        }

        /// <summary>EP message to <c>str2</c>.</summary>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str2</c> as the random arguments are varied. The formula is <c>proj[p(str2) sum_(str1,areEqual) p(str1,areEqual) factor(areEqual,str1,str2)]/p(str2)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static StringDistribution Str2AverageConditional(
            StringDistribution str1, [SkipIfUniform] Bernoulli areEqual, StringDistribution result)
        {
            return Str1AverageConditional(str1, areEqual, result);
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual) p(areEqual) factor(areEqual,str1,str2) / sum_areEqual p(areEqual) messageTo(areEqual))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str1,str2) p(str1,str2) factor(areEqual,str1,str2))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, StringDistribution str1, StringDistribution str2)
        {
            return LogAverageFactor(areEqual, str1, str2);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str1,str2) p(str1,str2) factor(areEqual,str1,str2))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, StringDistribution str1, StringDistribution str2)
        {
            Bernoulli messageToAreEqual = AreEqualAverageConditional(str1, str2);
            return messageToAreEqual.GetLogProb(areEqual);
        }

        #endregion
    }
}
