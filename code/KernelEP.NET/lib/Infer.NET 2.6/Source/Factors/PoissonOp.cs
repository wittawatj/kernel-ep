// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Poisson(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Poisson", typeof(double))]
    [Quality(QualityBand.Stable)]
    public static class PoissonOp
    {
        public static bool ForceProper;

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(int sample, double mean)
        {
            return SampleAverageConditional(mean).GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int sample, double mean)
        {
            return LogAverageFactor(sample, mean);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(int sample, double mean)
        {
            return LogAverageFactor(sample, mean);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,mean))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Poisson sample, double mean, [Fresh] Poisson to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,mean) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Poisson sample, double mean)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_mean">Outgoing message to <c>mean</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(mean) p(mean) factor(sample,mean))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(int sample, [SkipIfUniform] Gamma mean, [Fresh] Gamma to_mean)
        {
            return to_mean.GetLogAverageOf(mean);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="to_mean">Outgoing message to <c>mean</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(mean) p(mean) factor(sample,mean))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int sample, Gamma mean, [Fresh] Gamma to_mean)
        {
            return LogAverageFactor(sample, mean, to_mean);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,mean) p(sample,mean) factor(sample,mean))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Poisson sample, Gamma mean)
        {
            if (sample.IsUniform())
                return 0;
            if (sample.IsPointMass)
                return LogAverageFactor(sample.Point, mean, MeanAverageConditional(sample.Point));
            if (sample.Precision != 0)
                throw new NotImplementedException("sample.Precision != 0 is not implemented");
            return -mean.Shape * Math.Log(1 + (1 - sample.Rate) / mean.Rate) - sample.GetLogNormalizer();
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="to_sample">Previous outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,mean) p(sample,mean) factor(sample,mean) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Poisson sample, Gamma mean, Poisson to_sample)
        {
            return LogAverageFactor(sample, mean) - to_sample.GetLogAverageOf(sample);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <returns>The outgoing EP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma MeanAverageConditional(int sample)
        {
            // p(mean) = mean^sample exp(-mean)/Gamma(sample+1)
            return new Gamma(sample + 1, 1);
        }

        private const string NotSupportedMessage =
            "A Poisson factor with unobserved output is not yet implemented for Expectation Propagation.  Try using Variational Message Passing.";

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>mean</c> as the random arguments are varied. The formula is <c>proj[p(mean) sum_(sample) p(sample) factor(sample,mean)]/p(mean)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static Gamma MeanAverageConditional([SkipIfUniform] Poisson sample, [Proper] Gamma mean)
        {
            if (sample.IsUniform())
                return Gamma.Uniform();
            if (sample.IsPointMass)
                return MeanAverageConditional(sample.Point);
            if (sample.Precision != 0)
                throw new NotImplementedException("sample.Precision != 0 is not implemented");
            // Z = int_m sum_x r^x m^x exp(-m)/x! q(m)
            //   = int_m exp(rm -m) q(m)
            //   = (1 + (1-r)/b)^(-a)
            // logZ = -a log(1 + (1-r)/b)
            // alpha = -b dlogZ/db
            //       = -a(1-r)/(b + (1-r))
            // beta = -b dalpha/db
            //      = -ba(1-r)/(b + (1-r))^2
            double omr = 1 - sample.Rate;
            double denom = 1 / (mean.Rate + omr);
            double alpha = -mean.Shape * omr * denom;
            double beta = mean.Rate * alpha * denom;
            return GaussianOp.GammaFromAlphaBeta(mean, alpha, beta, ForceProper);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Poisson SampleAverageConditional(double mean)
        {
            return new Poisson(mean);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(mean) p(mean) factor(sample,mean)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static Poisson SampleAverageConditional(Poisson sample, [Proper] Gamma mean)
        {
            if (mean.IsPointMass)
                return SampleAverageConditional(mean.Point);
            if (sample.IsPointMass)
                return new Poisson(mean.GetMean());
            if (sample.Precision != 0)
                throw new NotImplementedException("sample.Precision != 0 is not implemented");
            // posterior mean of x is r dlogZ/dr = ar/(b + 1-r)
            // want to choose m such that rm = above
            return new Poisson(mean.Shape / (mean.Rate + 1 - sample.Rate));
        }

        //-- VMP -------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(mean) p(mean) log(factor(sample,mean))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(int sample, [Proper] Gamma mean)
        {
            return sample * mean.GetMeanLog() - MMath.GammaLn(sample + 1) - mean.GetMean();
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,mean) p(sample,mean) log(factor(sample,mean))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(Poisson sample, [Proper] Gamma mean)
        {
            return sample.GetMean() * mean.GetMeanLog() - sample.GetMeanLogFactorial() - mean.GetMean();
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,mean))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Poisson sample, double mean)
        {
            return sample.GetMean() * mean - sample.GetMeanLogFactorial() - mean;
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <returns>The outgoing VMP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma MeanAverageLogarithm(int sample)
        {
            return MeanAverageConditional(sample);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>mean</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,mean)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static Gamma MeanAverageLogarithm([Proper] Poisson sample)
        {
            // p(mean) = exp(E[sample]*log(mean) - mean - E[log(sample!)])
            return new Gamma(sample.GetMean() + 1, 1);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(mean) p(mean) log(factor(sample,mean)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static Poisson SampleAverageLogarithm([Proper] Gamma mean)
        {
            return new Poisson(Math.Exp(mean.GetMeanLog()));
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Poisson SampleAverageLogarithm(double mean)
        {
            return SampleAverageConditional(mean);
        }
    }
}
