/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.Concat(String, Char)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Concat", typeof(string), typeof(char))]
    [Quality(QualityBand.Experimental)]
    public static class StringCharConcatOp
    {
        #region EP messages

        /// <summary>EP message to <c>concat</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <returns>The outgoing EP message to the <c>concat</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>concat</c> as the random arguments are varied. The formula is <c>proj[p(concat) sum_(str,ch) p(str,ch) factor(concat,str,ch)]/p(concat)</c>.</para>
        /// </remarks>
        public static StringDistribution ConcatAverageConditional(StringDistribution str, DiscreteChar ch)
        {
            return StringConcatOp.ConcatAverageConditional(str, StringDistribution.Char(ch));
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(concat,ch) p(concat,ch) factor(concat,str,ch)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(StringDistribution concat, DiscreteChar ch)
        {
            return StringConcatOp.Str1AverageConditional(concat, StringDistribution.Char(ch));
        }

        /// <summary>EP message to <c>ch</c>.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>The outgoing EP message to the <c>ch</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ch</c> as the random arguments are varied. The formula is <c>proj[p(ch) sum_(concat,str) p(concat,str) factor(concat,str,ch)]/p(ch)</c>.</para>
        /// </remarks>
        public static DiscreteChar ChAverageConditional(StringDistribution concat, StringDistribution str)
        {
            var result = StringConcatOp.Str2AverageConditional(concat, str);
            return SingleOp.CharacterAverageConditional(result);
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(concat,str,ch) p(concat,str,ch) factor(concat,str,ch))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(StringDistribution concat, StringDistribution str, DiscreteChar ch)
        {
            return StringConcatOp.LogAverageFactor(concat, str, StringDistribution.Char(ch));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(concat,str,ch) p(concat,str,ch) factor(concat,str,ch) / sum_concat p(concat) messageTo(concat))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution concat, StringDistribution str, DiscreteChar ch)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Constant value for <c>concat</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str,ch) p(str,ch) factor(concat,str,ch))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(string concat, StringDistribution str, DiscreteChar ch)
        {
            return StringConcatOp.LogEvidenceRatio(concat, str, StringDistribution.Char(ch));
        }

        #endregion
    }
}
