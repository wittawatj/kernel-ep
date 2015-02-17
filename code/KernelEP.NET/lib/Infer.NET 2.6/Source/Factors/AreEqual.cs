// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.AreEqual(bool, bool)" />, given random arguments to the function.</summary>
    /// <remarks>This factor is symmetric among all three arguments.</remarks>
    [FactorMethod(typeof(Factor), "AreEqual", typeof(bool), typeof(bool))]
    [Quality(QualityBand.Mature)]
    public static class BooleanAreEqualOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, bool a, bool b)
        {
            return (areEqual == Factor.AreEqual(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, bool a, bool b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(bool areEqual, bool a, bool b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual) p(areEqual) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli areEqual, bool a, bool b)
        {
            return areEqual.GetLogProb(Factor.AreEqual(a, b));
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a,b) p(a,b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AreEqualAverageConditional([SkipIfUniform] Bernoulli A, [SkipIfUniform] Bernoulli B)
        {
            return Bernoulli.FromLogOdds(Bernoulli.LogitProbEqual(A.LogOdds, B.LogOdds));
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(b) p(b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AreEqualAverageConditional(bool A, [SkipIfUniform] Bernoulli B)
        {
            return Bernoulli.FromLogOdds(A ? B.LogOdds : -B.LogOdds);
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a) p(a) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Bernoulli AreEqualAverageConditional([SkipIfUniform] Bernoulli A, bool B)
        {
            return AreEqualAverageConditional(B, A);
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>areEqual</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(bool A, bool B)
        {
            return Bernoulli.PointMass(Factor.AreEqual(A, B));
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(areEqual,b) p(areEqual,b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(b) p(b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AAverageConditional(bool areEqual, [SkipIfUniform] Bernoulli B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(areEqual) p(areEqual) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Bernoulli AAverageConditional([SkipIfUniform] Bernoulli areEqual, bool B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli AAverageConditional(bool areEqual, bool B)
        {
            return AreEqualAverageConditional(areEqual, B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(areEqual,a) p(areEqual,a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a) p(a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageConditional(bool areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(areEqual) p(areEqual) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageConditional([SkipIfUniform] Bernoulli areEqual, bool A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli BAverageConditional(bool areEqual, bool A)
        {
            return AAverageConditional(areEqual, A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <param name="to_areEqual">Outgoing message to <c>areEqual</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual) p(areEqual) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli areEqual, [Fresh] Bernoulli to_areEqual)
        {
            return to_areEqual.GetLogAverageOf(areEqual);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, Bernoulli A, Bernoulli B, [Fresh] Bernoulli to_A)
        {
            //Bernoulli to_A = AAverageConditional(areEqual, B);
            return A.GetLogAverageOf(to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, Bernoulli A, bool B, [Fresh] Bernoulli to_A)
        {
            //Bernoulli toA = AAverageConditional(areEqual, B);
            return A.GetLogAverageOf(to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Outgoing message to <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, bool A, Bernoulli B, [Fresh] Bernoulli to_B)
        {
            return LogAverageFactor(areEqual, B, A, to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual) p(areEqual) factor(areEqual,a,b) / sum_areEqual p(areEqual) messageTo(areEqual))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, Bernoulli A, Bernoulli B, [Fresh] Bernoulli to_A)
        {
            return LogAverageFactor(areEqual, A, B, to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Outgoing message to <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, bool A, Bernoulli B, [Fresh] Bernoulli to_B)
        {
            return LogAverageFactor(areEqual, A, B, to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, Bernoulli A, bool B, [Fresh] Bernoulli to_A)
        {
            return LogEvidenceRatio(areEqual, B, A, to_A);
        }

        //- VMP -------------------------------------------------------------------------------------------------------------

        private const string NotSupportedMessage = "Variational Message Passing does not support an AreEqual factor with fixed output.";

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AreEqualAverageLogarithm([SkipIfUniform] Bernoulli A, [SkipIfUniform] Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(b) p(b) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AreEqualAverageLogarithm(bool A, [SkipIfUniform] Bernoulli B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(a) p(a) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Bernoulli AreEqualAverageLogarithm([SkipIfUniform] Bernoulli A, bool B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>areEqual</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageLogarithm(bool A, bool B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli B)
        {
            if (areEqual.IsPointMass)
                return AAverageLogarithm(areEqual.Point, B);
            // when AreEqual is marginalized, the factor is proportional to exp((A==B)*areEqual.LogOdds)
            return Bernoulli.FromLogOdds(areEqual.LogOdds * (2 * B.GetProbTrue() - 1));
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> with <c>areEqual</c> integrated out. The formula is <c>sum_areEqual p(areEqual) factor(areEqual,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Bernoulli AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, bool B)
        {
            return Bernoulli.FromLogOdds(B ? areEqual.LogOdds : -areEqual.LogOdds);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(BooleanAreEqualOp.NotSupportedMessage)]
        public static Bernoulli AAverageLogarithm(bool areEqual, [SkipIfUniform] Bernoulli B)
        {
            if (B.IsPointMass)
                return AAverageLogarithm(areEqual, B.Point);
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli AAverageLogarithm(bool areEqual, bool B)
        {
            return AAverageConditional(areEqual, B);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageLogarithm(areEqual, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>areEqual</c> integrated out. The formula is <c>sum_areEqual p(areEqual) factor(areEqual,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Bernoulli BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, bool A)
        {
            return AAverageLogarithm(areEqual, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [NotSupported(BooleanAreEqualOp.NotSupportedMessage)]
        public static Bernoulli BAverageLogarithm(bool areEqual, [SkipIfUniform] Bernoulli A)
        {
            return AAverageLogarithm(areEqual, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Bernoulli BAverageLogarithm(bool areEqual, bool A)
        {
            return AAverageLogarithm(areEqual, A);
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="EnumSupport.AreEqual{TEnum}(TEnum, TEnum)" /></description></item><item><description><see cref="Factor.AreEqual(int, int)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "AreEqual", typeof(int), typeof(int))]
    [FactorMethod(typeof(EnumSupport), "AreEqual<>")]
    [Quality(QualityBand.Mature)]
    public static class DiscreteAreEqualOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, int a, int b)
        {
            return (areEqual == Factor.AreEqual(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, int a, int b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(bool areEqual, int a, int b)
        {
            return LogAverageFactor(areEqual, a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual,a,b) p(areEqual,a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli areEqual, int a, int b)
        {
            return areEqual.GetLogProb(Factor.AreEqual(a, b));
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a,b) p(a,b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(Discrete A, Discrete B)
        {
            return Bernoulli.FromLogOdds(MMath.Logit(A.ProbEqual(B)));
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a,b) p(a,b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(int A, Discrete B)
        {
            return Bernoulli.FromLogOdds(MMath.Logit(B[A]));
        }

        /// <summary>EP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[p(areEqual) sum_(a,b) p(a,b) factor(areEqual,a,b)]/p(areEqual)</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageConditional(Discrete A, int B)
        {
            return AreEqualAverageConditional(B, A);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(areEqual,b) p(areEqual,b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete AAverageConditional([SkipIfUniform] Bernoulli areEqual, Discrete B, Discrete result)
        {
            if (areEqual.IsPointMass)
                return AAverageConditional(areEqual.Point, B, result);
            if (result == default(Discrete))
                result = Distributions.Discrete.Uniform(B.Dimension, B.Sparsity);
            double p = areEqual.GetProbTrue();
            Vector probs = result.GetWorkspace();
            probs = B.GetProbs(probs);
            probs.SetToProduct(probs, 2.0 * p - 1.0);
            probs.SetToSum(probs, 1.0 - p);
            result.SetProbs(probs);
            return result;
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(areEqual,b) p(areEqual,b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete AAverageConditional([SkipIfUniform] Bernoulli areEqual, int B, Discrete result)
        {
            if (areEqual.IsPointMass)
                return AAverageConditional(areEqual.Point, B, result);
            Vector probs = result.GetWorkspace();
            double p = areEqual.GetProbTrue();
            probs.SetAllElementsTo(1 - p);
            probs[B] = p;
            result.SetProbs(probs);
            return result;
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(b) p(b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        public static Discrete AAverageConditional(bool areEqual, Discrete B, Discrete result)
        {
            if (B.IsPointMass)
                return AAverageConditional(areEqual, B.Point, result);
            if (result == default(Discrete))
                result = Distributions.Discrete.Uniform(B.Dimension, B.Sparsity);
            if (areEqual)
                result.SetTo(B);
            else
            {
                Vector probs = result.GetWorkspace();
                probs = B.GetProbs(probs);
                probs.SetToDifference(1.0, probs);
                result.SetProbs(probs);
            }
            return result;
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(b) p(b) factor(areEqual,a,b)]/p(a)</c>.</para>
        /// </remarks>
        public static Discrete AAverageConditional(bool areEqual, int B, Discrete result)
        {
            if (areEqual)
                result.Point = B;
            else if (result.Dimension == 2)
                result.Point = 1 - B;
            else
            {
                Vector probs = result.GetWorkspace();
                probs.SetAllElementsTo(1);
                probs[B] = 0;
                result.SetProbs(probs);
            }
            return result;
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(areEqual,a) p(areEqual,a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete BAverageConditional([SkipIfUniform] Bernoulli areEqual, Discrete A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(areEqual,a) p(areEqual,a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete BAverageConditional([SkipIfUniform] Bernoulli areEqual, int A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a) p(a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static Discrete BAverageConditional(bool areEqual, Discrete A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a) p(a) factor(areEqual,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static Discrete BAverageConditional(bool areEqual, int A, Discrete result)
        {
            return AAverageConditional(areEqual, A, result);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <param name="to_areEqual">Outgoing message to <c>areEqual</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual) p(areEqual) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Bernoulli areEqual, [Fresh] Bernoulli to_areEqual)
        {
            return to_areEqual.GetLogAverageOf(areEqual);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, Discrete A, Discrete B)
        {
            Bernoulli to_areEqual = AreEqualAverageConditional(A, B);
            return to_areEqual.GetLogProb(areEqual);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, Discrete A, Discrete B, [Fresh] Discrete to_A)
        {
            return A.GetLogAverageOf(to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, int A, Discrete B)
        {
            Bernoulli to_areEqual = AreEqualAverageConditional(A, B);
            return to_areEqual.GetLogProb(areEqual);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Outgoing message to <c>B</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, int A, Discrete B, [Fresh] Discrete to_B)
        {
            return B.GetLogAverageOf(to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, Discrete A, int B)
        {
            Bernoulli to_areEqual = AreEqualAverageConditional(A, B);
            return to_areEqual.GetLogProb(areEqual);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(bool areEqual, Discrete A, int B, [Fresh] Discrete to_A)
        {
            return A.GetLogAverageOf(to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(areEqual) p(areEqual) factor(areEqual,a,b) / sum_areEqual p(areEqual) messageTo(areEqual))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Bernoulli areEqual)
        {
            return 0.0;
        }

        //public static double LogEvidenceRatio(bool areEqual, Discrete A, Discrete B) { return LogAverageFactor(areEqual, A, B); }
        //public static double LogEvidenceRatio(bool areEqual, int A, Discrete B) { return LogAverageFactor(areEqual, A, B); }
        //public static double LogEvidenceRatio(bool areEqual, Discrete A, int B) { return LogAverageFactor(areEqual, A, B); }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, Discrete A, Discrete B, [Fresh] Discrete to_A)
        {
            return LogAverageFactor(areEqual, A, B, to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Outgoing message to <c>B</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, int A, Discrete B, [Fresh] Discrete to_B)
        {
            return LogAverageFactor(areEqual, A, B, to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a,b) p(a,b) factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(bool areEqual, Discrete A, int B, [Fresh] Discrete to_A)
        {
            return LogAverageFactor(areEqual, A, B, to_A);
        }

        //-- VMP ----------------------------------------------------------------------------------------

        private const string NotSupportedMessage = "Variational Message Passing does not support an AreEqual factor with fixed output.";

        /// <summary>
        /// Evidence message for VMP.
        /// </summary>
        /// <returns>Zero</returns>
        /// <remarks><para>
        /// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
        /// </para></remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(areEqual,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageLogarithm(Discrete A, Discrete B)
        {
            // same as BP if you use John Winn's rule.
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageLogarithm(int A, Discrete B)
        {
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>areEqual</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>areEqual</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>areEqual</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(areEqual,a,b)]</c>.</para>
        /// </remarks>
        public static Bernoulli AreEqualAverageLogarithm(Discrete A, int B)
        {
            return AreEqualAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, Discrete B, Discrete result)
        {
            if (areEqual.IsPointMass)
                return AAverageLogarithm(areEqual.Point, B, result);
            if (result == default(Discrete))
                result = Discrete.Uniform(B.Dimension, B.Sparsity);
            // when AreEqual is marginalized, the factor is proportional to exp((A==B)*areEqual.LogOdds)
            Vector probs = result.GetWorkspace();
            probs = B.GetProbs(probs);
            probs.SetToFunction(probs, x => Math.Exp(x * areEqual.LogOdds));
            result.SetProbs(probs);
            return result;
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete AAverageLogarithm([SkipIfUniform] Bernoulli areEqual, int B, Discrete result)
        {
            return AAverageConditional(areEqual, B, result);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        [NotSupported(DiscreteAreEqualOp.NotSupportedMessage)]
        public static Discrete AAverageLogarithm(bool areEqual, Discrete B, Discrete result)
        {
            if (B.IsPointMass)
                return AAverageLogarithm(areEqual, B.Point, result);
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        public static Discrete AAverageLogarithm(bool areEqual, int B, Discrete result)
        {
            return AAverageConditional(areEqual, B, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, Discrete A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Incoming message from <c>areEqual</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>areEqual</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_areEqual p(areEqual) factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="areEqual" /> is not a proper distribution.</exception>
        public static Discrete BAverageLogarithm([SkipIfUniform] Bernoulli areEqual, int A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        [NotSupported(DiscreteAreEqualOp.NotSupportedMessage)]
        public static Discrete BAverageLogarithm(bool areEqual, Discrete A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="areEqual">Constant value for <c>areEqual</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(areEqual,a,b)))</c>.</para>
        /// </remarks>
        public static Discrete BAverageLogarithm(bool areEqual, int A, Discrete result)
        {
            return AAverageLogarithm(areEqual, A, result);
        }
    }
}
