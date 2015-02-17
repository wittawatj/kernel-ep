// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.BetaFromMeanAndTotalCount(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "BetaFromMeanAndTotalCount")]
    [Quality(QualityBand.Experimental)]
    public static class BetaOp
    {
        /// <summary>
        /// How much damping to use to avoid improper messages. A higher value implies more damping. 
        /// </summary>
        public static double damping = 0.0;

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(prob,mean,totalCount))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(double prob, double mean, double totalCount)
        {
            return LogAverageFactor(prob, mean, totalCount);
        }

        // TODO: VMP evidence messages for stochastic inputs (see DirichletOp)

        /// <summary>VMP message to <c>prob</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>prob</c>. The formula is <c>exp(sum_(mean,totalCount) p(mean,totalCount) log(factor(prob,mean,totalCount)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        public static Beta ProbAverageLogarithm(Beta mean, [Proper] Gamma totalCount)
        {
            double meanMean = mean.GetMean();
            double totalCountMean = totalCount.GetMean();
            return (new Beta(meanMean * totalCountMean, (1 - meanMean) * totalCountMean));
        }

        /// <summary>VMP message to <c>prob</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <returns>The outgoing VMP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>prob</c>. The formula is <c>exp(sum_(mean) p(mean) log(factor(prob,mean,totalCount)))</c>.</para>
        /// </remarks>
        public static Beta ProbAverageLogarithm(Beta mean, double totalCount)
        {
            double meanMean = mean.GetMean();
            return (new Beta(meanMean * totalCount, (1 - meanMean) * totalCount));
        }

        /// <summary>VMP message to <c>prob</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>prob</c>. The formula is <c>exp(sum_(totalCount) p(totalCount) log(factor(prob,mean,totalCount)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        public static Beta ProbAverageLogarithm(double mean, [Proper] Gamma totalCount)
        {
            double totalCountMean = totalCount.GetMean();
            return (new Beta(mean * totalCountMean, (1 - mean) * totalCountMean));
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_mean">Previous outgoing message to <c>mean</c>.</param>
        /// <returns>The outgoing VMP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>mean</c>. The formula is <c>exp(sum_(totalCount) p(totalCount) log(factor(prob,mean,totalCount)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        public static Beta MeanAverageLogarithm(double prob, Beta mean, [Proper] Gamma totalCount, Beta to_mean)
        {
            return MeanAverageLogarithm(Beta.PointMass(prob), mean, totalCount, to_mean);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <param name="to_mean">Previous outgoing message to <c>mean</c>.</param>
        /// <returns>The outgoing VMP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static Beta MeanAverageLogarithm(double prob, Beta mean, double totalCount, Beta to_mean)
        {
            return MeanAverageLogarithm(Beta.PointMass(prob), mean, Gamma.PointMass(totalCount), to_mean);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="prob">Incoming message from <c>prob</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_mean">Previous outgoing message to <c>mean</c>.</param>
        /// <returns>The outgoing VMP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>mean</c>. The formula is <c>exp(sum_(prob,totalCount) p(prob,totalCount) log(factor(prob,mean,totalCount)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="prob" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        public static Beta MeanAverageLogarithm([Proper] Beta prob, Beta mean, [Proper] Gamma totalCount, Beta to_mean)
        {
            // Calculate gradient using method for DirichletOp
            double ELogP, ELogOneMinusP;
            prob.GetMeanLogs(out ELogP, out ELogOneMinusP);
            Vector gradS = DirichletOp.CalculateGradientForMean(
                Vector.FromArray(new double[] { mean.TrueCount, mean.FalseCount }),
                totalCount,
                Vector.FromArray(new double[] { ELogP, ELogOneMinusP }));
            // Project onto a Beta distribution 
            Matrix A = new Matrix(2, 2);
            double c = MMath.Trigamma(mean.TotalCount);
            A[0, 0] = MMath.Trigamma(mean.TrueCount) - c;
            A[1, 0] = A[0, 1] = -c;
            A[1, 1] = MMath.Trigamma(mean.FalseCount) - c;
            Vector theta = GammaFromShapeAndRateOp.twoByTwoInverse(A) * gradS;
            Beta approximateFactor = new Beta(theta[0] + 1, theta[1] + 1);
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_mean ^ damping);
        }

        /// <summary>VMP message to <c>totalCount</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>.</param>
        /// <param name="prob">Incoming message from <c>prob</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_totalCount">Previous outgoing message to <c>totalCount</c>.</param>
        /// <returns>The outgoing VMP message to the <c>totalCount</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>totalCount</c>. The formula is <c>exp(sum_(mean,prob) p(mean,prob) log(factor(prob,mean,totalCount)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="prob" /> is not a proper distribution.</exception>
        public static Gamma TotalCountAverageLogarithm([Proper] Beta mean, Gamma totalCount, [SkipIfUniform] Beta prob, Gamma to_totalCount)
        {
            double ELogP, ELogOneMinusP;
            prob.GetMeanLogs(out ELogP, out ELogOneMinusP);
            Gamma approximateFactor = DirichletOp.TotalCountAverageLogarithmHelper(
                Vector.FromArray(new double[] { mean.TrueCount, mean.FalseCount }),
                totalCount,
                Vector.FromArray(new double[] { ELogP, ELogOneMinusP }));
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_totalCount ^ damping);
        }

        //---------------------------- EP -----------------------------

        private const string NotSupportedMessage = "Expectation Propagation does not currently support beta distributions with stochastic arguments.";

        /// <summary>Evidence message for EP.</summary>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(prob,mean,totalCount))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double prob, double mean, double totalCount)
        {
            var g = new Beta(mean * totalCount, (1 - mean) * totalCount);
            return g.GetLogProb(prob);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="prob">Incoming message from <c>prob</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(prob) p(prob) factor(prob,mean,totalCount))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Beta prob, double mean, double totalCount)
        {
            var g = new Beta(mean * totalCount, (1 - mean) * totalCount);
            return g.GetLogAverageOf(prob);
        }

        /// <summary>EP message to <c>prob</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <returns>The outgoing EP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>prob</c> conditioned on the given values.</para>
        /// </remarks>
        public static Beta ProbAverageConditional(double mean, double totalCount)
        {
            return new Beta(mean * totalCount, (1 - mean) * totalCount);
        }

        /// <summary>EP message to <c>prob</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>prob</c> as the random arguments are varied. The formula is <c>proj[p(prob) sum_(mean,totalCount) p(mean,totalCount) factor(prob,mean,totalCount)]/p(prob)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta ProbAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>prob</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <returns>The outgoing EP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>prob</c> as the random arguments are varied. The formula is <c>proj[p(prob) sum_(mean) p(mean) factor(prob,mean,totalCount)]/p(prob)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta ProbAverageConditional([SkipIfUniform] Beta mean, double totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>prob</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>prob</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>prob</c> as the random arguments are varied. The formula is <c>proj[p(prob) sum_(totalCount) p(totalCount) factor(prob,mean,totalCount)]/p(prob)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta ProbAverageConditional(double mean, [SkipIfUniform] Gamma totalCount)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>mean</c> as the random arguments are varied. The formula is <c>proj[p(mean) sum_(totalCount) p(totalCount) factor(prob,mean,totalCount)]/p(mean)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, double prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, double totalCount, double prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Constant value for <c>totalCount</c>.</param>
        /// <param name="prob">Incoming message from <c>prob</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>mean</c> as the random arguments are varied. The formula is <c>proj[p(mean) sum_(prob) p(prob) factor(prob,mean,totalCount)]/p(mean)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="prob" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, double totalCount, [SkipIfUniform] Beta prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="prob">Incoming message from <c>prob</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>mean</c> as the random arguments are varied. The formula is <c>proj[p(mean) sum_(totalCount,prob) p(totalCount,prob) factor(prob,mean,totalCount)]/p(mean)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="prob" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Beta MeanAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Beta prob, [SkipIfUniform] Beta result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>totalCount</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="prob">Constant value for <c>prob</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>totalCount</c> as the random arguments are varied. The formula is <c>proj[p(totalCount) sum_(mean) p(mean) factor(prob,mean,totalCount)]/p(totalCount)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, double prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>EP message to <c>totalCount</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="totalCount">Incoming message from <c>totalCount</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="prob">Incoming message from <c>prob</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>totalCount</c> as the random arguments are varied. The formula is <c>proj[p(totalCount) sum_(mean,prob) p(mean,prob) factor(prob,mean,totalCount)]/p(totalCount)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="totalCount" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="prob" /> is not a proper distribution.</exception>
        [NotSupported(NotSupportedMessage)]
        public static Gamma TotalCountAverageConditional([SkipIfUniform] Beta mean, [SkipIfUniform] Gamma totalCount, [SkipIfUniform] Beta prob, [SkipIfUniform] Gamma result)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Beta.Sample(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Beta), "Sample", typeof(double), typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class BetaFromTrueAndFalseCountsOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,trueCount,falseCount))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Beta sample, double trueCount, double falseCount, [Fresh] Beta to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,trueCount,falseCount) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Beta sample, double trueCount, double falseCount)
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,trueCount,falseCount))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Beta sample, double trueCount, double falseCount, [Fresh] Beta to_sample)
        {
            return to_sample.GetAverageLog(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,trueCount,falseCount))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double sample, double trueCount, double falseCount)
        {
            return SampleAverageConditional(trueCount, falseCount).GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,trueCount,falseCount))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double sample, double trueCount, double falseCount)
        {
            return LogAverageFactor(sample, trueCount, falseCount);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,trueCount,falseCount))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(double sample, double trueCount, double falseCount)
        {
            return LogAverageFactor(sample, trueCount, falseCount);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Beta SampleAverageLogarithm(double trueCount, double falseCount)
        {
            return new Beta(trueCount, falseCount);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="trueCount">Constant value for <c>trueCount</c>.</param>
        /// <param name="falseCount">Constant value for <c>falseCount</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Beta SampleAverageConditional(double trueCount, double falseCount)
        {
            return new Beta(trueCount, falseCount);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Beta.SampleFromMeanAndVariance(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(new string[] { "sample", "mean", "variance" }, typeof(Beta), "SampleFromMeanAndVariance")]
    [Quality(QualityBand.Stable)]
    public static class BetaFromMeanAndVarianceOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,mean,variance))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Beta sample, double mean, double variance, [Fresh] Beta to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,mean,variance) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Beta sample, double mean, double variance)
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Beta sample, double mean, double variance, [Fresh] Beta to_sample)
        {
            return to_sample.GetAverageLog(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,variance))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double sample, double mean, double variance)
        {
            return SampleAverageConditional(mean, variance).GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(double sample, double mean, double variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Beta SampleAverageLogarithm(double mean, double variance)
        {
            return Beta.FromMeanAndVariance(mean, variance);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Beta SampleAverageConditional(double mean, double variance)
        {
            return Beta.FromMeanAndVariance(mean, variance);
        }
    }
}
