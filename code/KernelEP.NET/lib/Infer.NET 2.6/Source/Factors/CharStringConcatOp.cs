/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.Concat(Char, String)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Concat", typeof(char), typeof(string))]
    [Quality(QualityBand.Experimental)]
    public static class CharStringConcatOp
    {
        #region EP messages

        /// <summary>EP message to <c>concat</c>.</summary>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>The outgoing EP message to the <c>concat</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>concat</c> as the random arguments are varied. The formula is <c>proj[p(concat) sum_(ch,str) p(ch,str) factor(concat,ch,str)]/p(concat)</c>.</para>
        /// </remarks>
        public static StringDistribution ConcatAverageConditional(DiscreteChar ch, StringDistribution str)
        {
            return StringConcatOp.ConcatAverageConditional(StringDistribution.Char(ch), str);
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(concat,ch) p(concat,ch) factor(concat,ch,str)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(StringDistribution concat, DiscreteChar ch)
        {
            return StringConcatOp.Str2AverageConditional(concat, StringDistribution.Char(ch));
        }

        /// <summary>EP message to <c>ch</c>.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>The outgoing EP message to the <c>ch</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ch</c> as the random arguments are varied. The formula is <c>proj[p(ch) sum_(concat,str) p(concat,str) factor(concat,ch,str)]/p(ch)</c>.</para>
        /// </remarks>
        public static DiscreteChar ChAverageConditional(StringDistribution concat, StringDistribution str)
        {
            var result = StringConcatOp.Str1AverageConditional(concat, str);
            return SingleOp.CharacterAverageConditional(result);
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(concat,ch,str) p(concat,ch,str) factor(concat,ch,str))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(StringDistribution concat, DiscreteChar ch, StringDistribution str)
        {
            return StringConcatOp.LogAverageFactor(concat, StringDistribution.Char(ch), str);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(concat,ch,str) p(concat,ch,str) factor(concat,ch,str) / sum_concat p(concat) messageTo(concat))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution concat, DiscreteChar ch, StringDistribution str)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Constant value for <c>concat</c>.</param>
        /// <param name="ch">Incoming message from <c>ch</c>.</param>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(ch,str) p(ch,str) factor(concat,ch,str))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(string concat, DiscreteChar ch, StringDistribution str)
        {
            return StringConcatOp.LogEvidenceRatio(concat, StringDistribution.Char(ch), str);
        }

        #endregion
    }
}
