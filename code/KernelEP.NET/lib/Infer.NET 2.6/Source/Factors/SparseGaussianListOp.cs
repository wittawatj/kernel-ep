// (C) Copyright 2011-2013 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;
    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="SparseGaussianList.Sample(ISparseList{double}, ISparseList{double})" />, given random arguments to the function.</summary>
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(SparseGaussianList), "Sample", typeof(ISparseList<double>), typeof(ISparseList<double>))]
    [Quality(QualityBand.Stable)]
    public static class SparseGaussianListOp
    {
        /// <summary />
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static SparseGaussianList SampleAverageConditionalInit([IgnoreDependency] ISparseList<double> mean)
        {
            return SparseGaussianList.FromSize(mean.Count);
        }

        /// <summary />
        /// <param name="mean">Incoming message from <c>means</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static SparseGaussianList SampleAverageConditionalInit([IgnoreDependency] SparseGaussianList mean)
        {
            return (SparseGaussianList)mean.Clone();
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseGaussianList SampleAverageConditional(ISparseList<double> mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageConditional(m, p));
            return result;
        }

        /// <summary>EP message to <c>means</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>means</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseGaussianList MeanAverageConditional(ISparseList<double> sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageConditional(s, p));
            return result;
        }

        /// <summary>EP message to <c>precs</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>precs</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseGammaList PrecisionAverageConditional(ISparseList<double> sample, ISparseList<double> mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageConditional(s, m));
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(means) p(means) factor(sample,means,precs)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static SparseGaussianList SampleAverageConditional([SkipIfUniform] SparseGaussianList mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageConditional(m, p));
            return result;
        }

        /// <summary>EP message to <c>means</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>means</c> as the random arguments are varied. The formula is <c>proj[p(means) sum_(sample) p(sample) factor(sample,means,precs)]/p(means)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static SparseGaussianList MeanAverageConditional([SkipIfUniform] SparseGaussianList sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageConditional(s, p));
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(means,precs) p(means,precs) factor(sample,means,precs)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList SampleAverageConditional(
            SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.SampleAverageConditional(s, m, p, tp));
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(precs) p(precs) factor(sample,means,precs)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList SampleAverageConditional(
            SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.SampleAverageConditional(s, m, p, tp));
            return result;
        }

        /// <summary>EP message to <c>means</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>means</c> as the random arguments are varied. The formula is <c>proj[p(means) sum_(sample,precs) p(sample,precs) factor(sample,means,precs)]/p(means)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList MeanAverageConditional(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.MeanAverageConditional(s, m, p, tp));
            return result;
        }

        /// <summary>EP message to <c>means</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>means</c> as the random arguments are varied. The formula is <c>proj[p(means) sum_(precs) p(precs) factor(sample,means,precs)]/p(means)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList MeanAverageConditional(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, mean, precision, to_precision, (s, m, p, tp) => GaussianOp.MeanAverageConditional(s, m, p, tp));
            return result;
        }

        /// <summary>EP message to <c>precs</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>precs</c> as the random arguments are varied. The formula is <c>proj[p(precs) sum_(means) p(means) factor(sample,means,precs)]/p(precs)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGammaList PrecisionAverageConditional(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, precision, (s, m, p) => GaussianOp.PrecisionAverageConditional_slow(Gaussian.PointMass(s), m, p));
            return result;
        }

        /// <summary>EP message to <c>precs</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>precs</c> as the random arguments are varied. The formula is <c>proj[p(precs) sum_(sample) p(sample) factor(sample,means,precs)]/p(precs)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGammaList PrecisionAverageConditional(
            [SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, precision, (s, m, p) => GaussianOp.PrecisionAverageConditional_slow(s, Gaussian.PointMass(m), p));
            return result;
        }

        /// <summary>EP message to <c>precs</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>precs</c> as the random arguments are varied. The formula is <c>proj[p(precs) sum_(sample,means) p(sample,means) factor(sample,means,precs)]/p(precs)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGammaList PrecisionAverageConditional(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, precision, (s, m, p) => GaussianOp.PrecisionAverageConditional_slow(s, m, p));
            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,means,precs))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(ISparseList<double> sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<double, double, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,means) p(sample,means) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<Gaussian, Gaussian, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<Gaussian, double, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(means) p(means) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<double, Gaussian, double, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(precs) p(precs) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(ISparseList<double> sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision)
        {
            Func<double, double, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(means,precs) p(means,precs) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<double, Gaussian, Gamma, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,precs) p(sample,precs) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(
            [SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<Gaussian, double, Gamma, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,means,precs) p(sample,means,precs) factor(sample,means,precs))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<Gaussian, Gaussian, Gamma, Gamma, double> f = GaussianOp.LogAverageFactor;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(ISparseList<double> sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<double, double, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,means) p(sample,means) factor(sample,means,precs) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(SparseGaussianList sample, SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<Gaussian, Gaussian, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,means,precs) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(SparseGaussianList sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<Gaussian, double, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(means) p(means) factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(ISparseList<double> sample, SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<double, Gaussian, double, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(precs) p(precs) factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(ISparseList<double> sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision)
        {
            Func<double, double, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(means,precs) p(means,precs) factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(
            ISparseList<double> sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, SparseGammaList to_precision)
        {
            Func<double, Gaussian, Gamma, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision, to_precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,means,precs) p(sample,means,precs) factor(sample,means,precs) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(
            [SkipIfUniform] SparseGaussianList sample, [SkipIfUniform] SparseGaussianList mean, [SkipIfUniform] SparseGammaList precision, [Fresh] SparseGaussianList to_sample, SparseGammaList to_precision)
        {
            Func<Gaussian, Gaussian, Gamma, Gaussian, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision, to_sample, to_precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,precs) p(sample,precs) factor(sample,means,precs) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(
            [SkipIfUniform] SparseGaussianList sample, ISparseList<double> mean, [SkipIfUniform] SparseGammaList precision, [Fresh] SparseGaussianList to_sample, SparseGammaList to_precision)
        {
            Func<Gaussian, double, Gamma, Gaussian, Gamma, double> f = GaussianOp.LogEvidenceRatio;
            return f.Map(sample, mean, precision, to_sample, to_precision).EnumerableSum(x => x);
        }

        //-- VMP ------------------------------------------------------------------------------------------------

        /// <summary />
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static SparseGaussianList SampleAverageLogarithmInit([IgnoreDependency] ISparseList<double> mean)
        {
            return SparseGaussianList.FromSize(mean.Count);
        }

        /// <summary />
        /// <param name="mean">Incoming message from <c>means</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static SparseGaussianList SampleAverageLogarithmInit([IgnoreDependency] SparseGaussianList mean)
        {
            return (SparseGaussianList)mean.Clone();
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseGaussianList SampleAverageLogarithm(ISparseList<double> mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(means) p(means) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static SparseGaussianList SampleAverageLogarithm([Proper] SparseGaussianList mean, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <summary>VMP message to <c>means</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>means</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseGaussianList MeanAverageLogarithm(ISparseList<double> sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <summary>VMP message to <c>means</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>means</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static SparseGaussianList MeanAverageLogarithm([Proper] SparseGaussianList sample, ISparseList<double> precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(means,precs) p(means,precs) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList SampleAverageLogarithm([Proper] SparseGaussianList mean, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <summary>VMP message to <c>means</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>means</c>. The formula is <c>exp(sum_(sample,precs) p(sample,precs) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList MeanAverageLogarithm([Proper] SparseGaussianList sample, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(precs) p(precs) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList SampleAverageLogarithm(ISparseList<double> mean, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(mean, precision, (m, p) => GaussianOp.SampleAverageLogarithm(m, p));
            return result;
        }

        /// <summary>VMP message to <c>means</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>means</c>. The formula is <c>exp(sum_(precs) p(precs) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static SparseGaussianList MeanAverageLogarithm(ISparseList<double> sample, [Proper] SparseGammaList precision, SparseGaussianList result)
        {
            result.SetToFunction(sample, precision, (s, p) => GaussianOp.MeanAverageLogarithm(s, p));
            return result;
        }

        /// <summary>VMP message to <c>precs</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>precs</c> conditioned on the given values.</para>
        /// </remarks>
        public static SparseGammaList PrecisionAverageLogarithm(ISparseList<double> sample, ISparseList<double> mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <summary>VMP message to <c>precs</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>precs</c>. The formula is <c>exp(sum_(sample,means) p(sample,means) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static SparseGammaList PrecisionAverageLogarithm([Proper] SparseGaussianList sample, [Proper] SparseGaussianList mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <summary>VMP message to <c>precs</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>precs</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static SparseGammaList PrecisionAverageLogarithm([Proper] SparseGaussianList sample, ISparseList<double> mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <summary>VMP message to <c>precs</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>precs</c>. The formula is <c>exp(sum_(means) p(means) log(factor(sample,means,precs)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static SparseGammaList PrecisionAverageLogarithm(ISparseList<double> sample, [Proper] SparseGaussianList mean, SparseGammaList result)
        {
            result.SetToFunction(sample, mean, (s, m) => GaussianOp.PrecisionAverageLogarithm(s, m));
            return result;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,means,precs) p(sample,means,precs) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, [Proper] SparseGaussianList mean, [Proper] SparseGammaList precision)
        {
            Func<Gaussian, Gaussian, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(precs) p(precs) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(ISparseList<double> sample, ISparseList<double> mean, [Proper] SparseGammaList precision)
        {
            Func<double, double, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(ISparseList<double> sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<double, double, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, ISparseList<double> mean, ISparseList<double> precision)
        {
            Func<Gaussian, double, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(means) p(means) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(ISparseList<double> sample, [Proper] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<double, Gaussian, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(means,precs) p(means,precs) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(ISparseList<double> sample, [Proper] SparseGaussianList mean, [Proper] SparseGammaList precision)
        {
            Func<double, Gaussian, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Constant value for <c>means</c>.</param>
        /// <param name="precision">Incoming message from <c>precs</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,precs) p(sample,precs) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, ISparseList<double> mean, [Proper] SparseGammaList precision)
        {
            Func<Gaussian, double, Gamma, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>means</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precs</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,means) p(sample,means) log(factor(sample,means,precs))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([Proper] SparseGaussianList sample, [Proper] SparseGaussianList mean, ISparseList<double> precision)
        {
            Func<Gaussian, Gaussian, double, double> f = GaussianOp.AverageLogFactor;
            return f.Map(sample, mean, precision).EnumerableSum(x => x);
        }
    }
}
