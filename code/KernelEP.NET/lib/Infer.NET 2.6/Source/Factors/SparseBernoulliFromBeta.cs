// (C) Copyright 2009-2013 Microsoft Research Cambridge
namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;
    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// <summary>Provides outgoing messages for <see cref="SparseBernoulliList.Sample(ISparseList{double})" />, given random arguments to the function.</summary>
    /// </summary>
    [FactorMethod(typeof(SparseBernoulliList), "Sample", typeof(ISparseList<double>))]
    [Quality(QualityBand.Stable)]
    public class SparseBernoulliFromBetaOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,probTrue))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(ISparseList<bool> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,probTrue))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(SparseBernoulliList sample, [Fresh] SparseBernoulliList to_sample)
        {
            Func<Bernoulli, Bernoulli, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample, to_sample).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(probTrue) p(probTrue) factor(sample,probTrue))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(ISparseList<bool> sample, SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.LogAverageFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,probTrue) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(SparseBernoulliList sample, ISparseList<double> probTrue)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(probTrue) p(probTrue) factor(sample,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(ISparseList<bool> sample, SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.LogEvidenceRatio;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(ISparseList<bool> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.LogEvidenceRatio;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,probTrue) p(sample,probTrue) factor(sample,probTrue) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(SparseBernoulliList sample, SparseBetaList probTrue)
        {
            return 0.0;
        }

        /// <summary>Gibbs message to <c>sample</c>.</summary>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseBernoulliList SampleConditional(ISparseList<double> probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleConditional(pt));
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseBernoulliList SampleAverageConditional(ISparseList<double> probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageConditional(pt));
            return result;
        }

        /// <summary>Gibbs message to <c>probTrue</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>probTrue</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseBetaList ProbTrueConditional(ISparseList<bool> sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueConditional(s));
            return result;
        }

        /// <summary>EP message to <c>probTrue</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>probTrue</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseBetaList ProbTrueAverageConditional(ISparseList<bool> sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueAverageConditional(s));
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(probTrue) p(probTrue) factor(sample,probTrue)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probTrue" /> is not a proper distribution.</exception>
        public static SparseBernoulliList SampleAverageConditional([SkipIfUniform] SparseBetaList probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageConditional(pt));
            return result;
        }

        /// <summary>EP message to <c>probTrue</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>probTrue</c> as the random arguments are varied. The formula is <c>proj[p(probTrue) sum_(sample) p(sample) factor(sample,probTrue)]/p(probTrue)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static SparseBetaList ProbTrueAverageConditional([SkipIfUniform] SparseBernoulliList sample, SparseBetaList probTrue, SparseBetaList result)
        {
            result.SetToFunction(sample, probTrue, (s, pt) => BernoulliFromBetaOp.ProbTrueAverageConditional(s, pt));
            return result;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,probTrue) p(sample,probTrue) log(factor(sample,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probTrue" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(SparseBernoulliList sample, [Proper] SparseBetaList probTrue)
        {
            Func<Bernoulli, Beta, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(SparseBernoulliList sample, ISparseList<double> probTrue)
        {
            Func<Bernoulli, double, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(probTrue) p(probTrue) log(factor(sample,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probTrue" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(ISparseList<bool> sample, [Proper] SparseBetaList probTrue)
        {
            Func<bool, Beta, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,probTrue))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(ISparseList<bool> sample, ISparseList<double> probTrue)
        {
            Func<bool, double, double> f = BernoulliFromBetaOp.AverageLogFactor;
            return f.Map(sample, probTrue).EnumerableSum(x => x);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="probTrue">Constant value for <c>probTrue</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseBernoulliList SampleAverageLogarithm(ISparseList<double> probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageLogarithm(pt));
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="probTrue">Incoming message from <c>probTrue</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(probTrue) p(probTrue) log(factor(sample,probTrue)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="probTrue" /> is not a proper distribution.</exception>
        public static SparseBernoulliList SampleAverageLogarithm([SkipIfUniform] SparseBetaList probTrue, SparseBernoulliList result)
        {
            result.SetToFunction(probTrue, pt => BernoulliFromBetaOp.SampleAverageLogarithm(pt));
            return result;
        }

        /// <summary>VMP message to <c>probTrue</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>probTrue</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseBetaList ProbTrueAverageLogarithm(ISparseList<bool> sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueAverageLogarithm(s));
            return result;
        }

        /// <summary>VMP message to <c>probTrue</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>probTrue</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,probTrue)))</c>.</para>
        /// </remarks>
        public static SparseBetaList ProbTrueAverageLogarithm(SparseBernoulliList sample, SparseBetaList result)
        {
            result.SetToFunction(sample, s => BernoulliFromBetaOp.ProbTrueAverageLogarithm(s));
            return result;
        }
    }
}
