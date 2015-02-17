// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.Not(bool)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Not")]
    [Quality(QualityBand.Mature)]
    public static class BooleanNotOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(not,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool not, bool b)
        {
            return (not == Factor.Not(b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(not,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool not, bool b)
        {
            return LogAverageFactor(not, b);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(not,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(bool not, bool b)
        {
            return LogAverageFactor(not, b);
        }

        /// <summary>EP message to <c>not</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>not</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>not</c> as the random arguments are varied. The formula is <c>proj[p(not) sum_(b) p(b) factor(not,b)]/p(not)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="b" /> is not a proper distribution.</exception>
        public static Bernoulli NotAverageConditional([SkipIfUniform] Bernoulli b)
        {
            return Bernoulli.FromLogOdds(-b.LogOdds);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="not">Incoming message from <c>not</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(not) p(not) factor(not,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="not" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli not)
        {
            return Bernoulli.FromLogOdds(-not.LogOdds);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli BAverageConditional(bool not)
        {
            return Bernoulli.PointMass(!not);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Incoming message from <c>not</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <param name="to_not">Outgoing message to <c>not</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(not,b) p(not,b) factor(not,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli not, Bernoulli b, [Fresh] Bernoulli to_not)
        {
            return to_not.GetLogAverageOf(not);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Incoming message from <c>not</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(not) p(not) factor(not,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli not, bool b)
        {
            return LogAverageFactor(b, not);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(not,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool not, Bernoulli b)
        {
            return not ? b.GetLogProbFalse() : b.GetLogProbTrue();
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Incoming message from <c>not</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(not) p(not) factor(not,b) / sum_not p(not) messageTo(not))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli not)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(not,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool not, Bernoulli b)
        {
            return LogAverageFactor(not, b);
        }


        //- VMP ---------------------------------------------------------------------------

        /// <summary>VMP message to <c>not</c>.</summary>
        /// <param name="b">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>not</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>not</c> as the random arguments are varied. The formula is <c>proj[sum_(b) p(b) factor(not,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="b" /> is not a proper distribution.</exception>
        public static Bernoulli NotAverageLogarithm([SkipIfUniform] Bernoulli b)
        {
            return NotAverageConditional(b);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="not">Incoming message from <c>not</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>not</c> integrated out. The formula is <c>sum_not p(not) factor(not,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="not" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli not)
        {
            return BAverageConditional(not);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="not">Constant value for <c>not</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli BAverageLogarithm(bool not)
        {
            return BAverageConditional(not);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(not,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }
    }
}
