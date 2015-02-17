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
    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.Concat(String, String)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Concat", typeof(string), typeof(string))]
    [Quality(QualityBand.Experimental)]
    public static class StringConcatOp
    {
        #region EP messages

        /// <summary>EP message to <c>concat</c>.</summary>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>The outgoing EP message to the <c>concat</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>concat</c> as the random arguments are varied. The formula is <c>proj[p(concat) sum_(str1,str2) p(str1,str2) factor(concat,str1,str2)]/p(concat)</c>.</para>
        /// </remarks>
        public static StringDistribution ConcatAverageConditional(StringDistribution str1, StringDistribution str2)
        {
            Argument.CheckIfNotNull(str1, "str1");
            Argument.CheckIfNotNull(str2, "str2");
            
            return str1 + str2;
        }

        /// <summary>EP message to <c>str1</c>.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>The outgoing EP message to the <c>str1</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str1</c> as the random arguments are varied. The formula is <c>proj[p(str1) sum_(concat,str2) p(concat,str2) factor(concat,str1,str2)]/p(str1)</c>.</para>
        /// </remarks>
        public static StringDistribution Str1AverageConditional(StringDistribution concat, StringDistribution str2)
        {
            Argument.CheckIfNotNull(concat, "concat");
            Argument.CheckIfNotNull(str2, "str2");
            
            StringTransducer transducer = StringTransducer.Copy();
            transducer.AppendInPlace(StringTransducer.Consume(str2.GetProbabilityFunction()));
            return StringDistribution.FromWorkspace(transducer.ProjectSource(concat));
        }

        /// <summary>EP message to <c>str2</c>.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <returns>The outgoing EP message to the <c>str2</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str2</c> as the random arguments are varied. The formula is <c>proj[p(str2) sum_(concat,str1) p(concat,str1) factor(concat,str1,str2)]/p(str2)</c>.</para>
        /// </remarks>
        public static StringDistribution Str2AverageConditional(StringDistribution concat, StringDistribution str1)
        {
            Argument.CheckIfNotNull(concat, "concat");
            Argument.CheckIfNotNull(str1, "str1");

            StringTransducer transducer = StringTransducer.Consume(str1.GetProbabilityFunction());
            transducer.AppendInPlace(StringTransducer.Copy());
            return StringDistribution.FromWorkspace(transducer.ProjectSource(concat));
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(concat,str1,str2) p(concat,str1,str2) factor(concat,str1,str2))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(StringDistribution concat, StringDistribution str1, StringDistribution str2)
        {
            Argument.CheckIfNotNull(concat, "concat");
            
            StringDistribution messageToConcat = ConcatAverageConditional(str1, str2);
            return messageToConcat.GetLogAverageOf(concat);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Incoming message from <c>concat</c>.</param>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(concat,str1,str2) p(concat,str1,str2) factor(concat,str1,str2) / sum_concat p(concat) messageTo(concat))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution concat, StringDistribution str1, StringDistribution str2)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="concat">Constant value for <c>concat</c>.</param>
        /// <param name="str1">Incoming message from <c>str1</c>.</param>
        /// <param name="str2">Incoming message from <c>str2</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str1,str2) p(str1,str2) factor(concat,str1,str2))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(string concat, StringDistribution str1, StringDistribution str2)
        {
            Argument.CheckIfNotNull(concat, "concat");
            
            return LogAverageFactor(StringDistribution.String(concat), str1, str2);
        }

        #endregion
    }
}
