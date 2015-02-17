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
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.Substring(String, int, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Substring")]
    [Quality(QualityBand.Experimental)]
    public static class SubstringOp
    {
        #region EP messages

        /// <summary>EP message to <c>sub</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>The outgoing EP message to the <c>sub</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sub</c> as the random arguments are varied. The formula is <c>proj[p(sub) sum_(str) p(str) factor(sub,str,start,length)]/p(sub)</c>.</para>
        /// </remarks>
        public static StringDistribution SubAverageConditional(StringDistribution str, int start, int length)
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");

            if (str.IsPointMass)
            {
                return SubAverageConditional(str.Point, start, length);
            }

            var anyChar = StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Any());
            var transducer = StringTransducer.Consume(StringAutomaton.Repeat(anyChar, minTimes: start, maxTimes: start));
            transducer.AppendInPlace(StringTransducer.Copy(StringAutomaton.Repeat(anyChar, minTimes: length, maxTimes: length)));
            transducer.AppendInPlace(StringTransducer.Consume(StringAutomaton.Constant(1.0)));
            
            return StringDistribution.FromWorkspace(transducer.ProjectSource(str));
        }

        /// <summary>EP message to <c>sub</c>.</summary>
        /// <param name="str">Constant value for <c>str</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>The outgoing EP message to the <c>sub</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sub</c> conditioned on the given values.</para>
        /// </remarks>
        public static StringDistribution SubAverageConditional(string str, int start, int length)
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");

            if (start + length > str.Length)
            {
                return StringDistribution.Zero();
            }
            
            return StringDistribution.String(str.Substring(start, length));
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="sub">Constant value for <c>sub</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>str</c> conditioned on the given values.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(string sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");
            
            if (sub.Length != length)
            {
                return StringDistribution.Zero();
            }

            StringDistribution result = StringDistribution.Any(minLength: start, maxLength: start);
            result.AppendInPlace(sub);
            result.AppendInPlace(StringDistribution.Any());
            return result;
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="sub">Incoming message from <c>sub</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(sub) p(sub) factor(sub,str,start,length)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(StringDistribution sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            Argument.CheckIfInRange(start >= 0, "start", "Start index must be non-negative.");
            Argument.CheckIfInRange(length >= 0, "length", "Length must be non-negative.");

            var anyChar = StringAutomaton.ConstantOnElement(1.0, DiscreteChar.Any());
            var transducer = StringTransducer.Produce(StringAutomaton.Repeat(anyChar, minTimes: start, maxTimes: start));
            transducer.AppendInPlace(StringTransducer.Copy(StringAutomaton.Repeat(anyChar, minTimes: length, maxTimes: length)));
            transducer.AppendInPlace(StringTransducer.Produce(StringAutomaton.Constant(1.0)));

            return StringDistribution.FromWorkspace(transducer.ProjectSource(sub));
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="sub">Incoming message from <c>sub</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str,sub) p(str,sub) factor(sub,str,start,length))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(StringDistribution str, StringDistribution sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            
            StringDistribution messageToSub = SubAverageConditional(str, start, length);
            return messageToSub.GetLogAverageOf(sub);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="sub">Incoming message from <c>sub</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str,sub) p(str,sub) factor(sub,str,start,length) / sum_sub p(sub) messageTo(sub))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution str, StringDistribution sub, int start, int length)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="sub">Constant value for <c>sub</c>.</param>
        /// <param name="start">Constant value for <c>start</c>.</param>
        /// <param name="length">Constant value for <c>length</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str) p(str) factor(sub,str,start,length))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(StringDistribution str, string sub, int start, int length)
        {
            Argument.CheckIfNotNull(sub, "sub");
            
            return LogAverageFactor(str, StringDistribution.String(sub), start, length);
        }

        #endregion
    }
}
