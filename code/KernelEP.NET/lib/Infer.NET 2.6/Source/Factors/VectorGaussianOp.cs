// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="VectorGaussian.Sample(Vector, PositiveDefiniteMatrix)" /></description></item><item><description><see cref="Factor.VectorGaussian(Vector, PositiveDefiniteMatrix)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(VectorGaussian), "Sample", typeof(Vector), typeof(PositiveDefiniteMatrix))]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "VectorGaussian")]
    [Buffers("SampleMean", "SampleVariance", "MeanMean", "MeanVariance", "PrecisionMean", "PrecisionMeanLogDet")]
    [Quality(QualityBand.Stable)]
    public static class VectorGaussianOp
    {
        /// <summary>Initialize the buffer <c>SampleVariance</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>.</param>
        /// <returns>Initial value of buffer <c>SampleVariance</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static PositiveDefiniteMatrix SampleVarianceInit([IgnoreDependency] VectorGaussian Sample)
        {
            return new PositiveDefiniteMatrix(Sample.Dimension, Sample.Dimension);
        }

        /// <summary>Update the buffer <c>SampleVariance</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        [Fresh]
        public static PositiveDefiniteMatrix SampleVariance([Proper] VectorGaussian Sample, PositiveDefiniteMatrix result)
        {
            return Sample.GetVariance(result);
        }

        /// <summary>Initialize the buffer <c>SampleMean</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>.</param>
        /// <returns>Initial value of buffer <c>SampleMean</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Vector SampleMeanInit([IgnoreDependency] VectorGaussian Sample)
        {
            return Vector.Zero(Sample.Dimension);
        }

        /// <summary>Update the buffer <c>SampleMean</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        [Fresh]
        public static Vector SampleMean([Proper] VectorGaussian Sample, [Fresh] PositiveDefiniteMatrix SampleVariance, Vector result)
        {
            return Sample.GetMean(result, SampleVariance);
        }

        /// <summary>Initialize the buffer <c>MeanVariance</c>.</summary>
        /// <param name="Mean">Incoming message from <c>mean</c>.</param>
        /// <returns>Initial value of buffer <c>MeanVariance</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static PositiveDefiniteMatrix MeanVarianceInit([IgnoreDependency] VectorGaussian Mean)
        {
            return new PositiveDefiniteMatrix(Mean.Dimension, Mean.Dimension);
        }

        /// <summary>Update the buffer <c>MeanVariance</c>.</summary>
        /// <param name="Mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Mean" /> is not a proper distribution.</exception>
        [Fresh]
        public static PositiveDefiniteMatrix MeanVariance([Proper] VectorGaussian Mean, PositiveDefiniteMatrix result)
        {
            return Mean.GetVariance(result);
        }

        /// <summary>Initialize the buffer <c>MeanMean</c>.</summary>
        /// <param name="Mean">Incoming message from <c>mean</c>.</param>
        /// <returns>Initial value of buffer <c>MeanMean</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Vector MeanMeanInit([IgnoreDependency] VectorGaussian Mean)
        {
            return Vector.Zero(Mean.Dimension);
        }

        /// <summary>Update the buffer <c>MeanMean</c>.</summary>
        /// <param name="Mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Mean" /> is not a proper distribution.</exception>
        [Fresh]
        public static Vector MeanMean([Proper] VectorGaussian Mean, [Fresh] PositiveDefiniteMatrix MeanVariance, Vector result)
        {
            return Mean.GetMean(result, MeanVariance);
        }

        /// <summary>Initialize the buffer <c>PrecisionMean</c>.</summary>
        /// <param name="Precision">Incoming message from <c>precision</c>.</param>
        /// <returns>Initial value of buffer <c>PrecisionMean</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static PositiveDefiniteMatrix PrecisionMeanInit([IgnoreDependency] Wishart Precision)
        {
            return new PositiveDefiniteMatrix(Precision.Dimension, Precision.Dimension);
        }

        /// <summary>Update the buffer <c>PrecisionMean</c>.</summary>
        /// <param name="Precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Precision" /> is not a proper distribution.</exception>
        [Fresh]
        public static PositiveDefiniteMatrix PrecisionMean([Proper] Wishart Precision, PositiveDefiniteMatrix result)
        {
            return Precision.GetMean(result);
        }

        /// <summary>Update the buffer <c>PrecisionMeanLogDet</c>.</summary>
        /// <param name="Precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>New value of buffer <c>PrecisionMeanLogDet</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Precision" /> is not a proper distribution.</exception>
        [Fresh]
        public static double PrecisionMeanLogDet([Proper] Wishart Precision)
        {
            return Precision.GetMeanLogDeterminant();
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,precision))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            int Dimension = sample.Count;
            LowerTriangularMatrix precL = new LowerTriangularMatrix(Dimension, Dimension);
            Vector iLb = Vector.Zero(Dimension);
            Vector precisionTimesMean = precision * mean;
            return VectorGaussian.GetLogProb(sample, precisionTimesMean, precision, precL, iLb);
        }

        /// <summary>Gibbs message to <c>sample</c>.</summary>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SampleConditional(Vector Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            result.SetMeanAndPrecision(Mean, Precision);
            return result;
        }

        /// <summary>Gibbs message to <c>mean</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian MeanConditional(Vector Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Sample, Precision, result);
        }

        /// <summary>Gibbs message to <c>precision</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <param name="diff" />
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>precision</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart PrecisionConditional(Vector Sample, Vector Mean, Wishart result, Vector diff)
        {
            if (result == default(Wishart))
                result = new Wishart(Sample.Count);
            diff.SetToDifference(Sample, Mean);
            const double SQRT_HALF = 0.70710678118654752440084436210485;
            diff.Scale(SQRT_HALF);
            result.Rate.SetToOuter(diff, diff);
            result.Shape = 0.5 * (result.Dimension + 2);
            return result;
        }

        /// <summary>Gibbs message to <c>precision</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>precision</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart PrecisionConditional(Vector Sample, Vector Mean, Wishart result)
        {
            Vector workspace = Vector.Zero(Sample.Count);
            return PrecisionConditional(Sample, Mean, result, workspace);
        }

        //-- EP -----------------------------------------------------------------------------------------------

        /// <summary>Evidence message for EP.</summary>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,precision))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            PositiveDefiniteMatrix Precision)
        {
            return VectorGaussian.GetLogProb(SampleMean, MeanMean, Precision.Inverse() + SampleVariance + MeanVariance);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,precision))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(
            Vector Sample, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, PositiveDefiniteMatrix Precision)
        {
            return VectorGaussian.GetLogProb(Sample, MeanMean, Precision.Inverse() + MeanVariance);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            return LogAverageFactor(sample, mean, precision);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(mean) p(mean) factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(
            Vector sample, VectorGaussian mean, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, PositiveDefiniteMatrix precision)
        {
            return LogAverageFactor(sample, MeanMean, MeanVariance, precision);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,mean,precision) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,mean) p(sample,mean) factor(sample,mean,precision) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] VectorGaussian sample, [SkipIfUniform] VectorGaussian mean, PositiveDefiniteMatrix precision)
        {
            return 0.0;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SampleAverageConditional(Vector Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Mean, Precision, result);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="Mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(mean) p(mean) factor(sample,mean,precision)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Mean" /> is not a proper distribution.</exception>
        public static VectorGaussian SampleAverageConditional([SkipIfUniform] VectorGaussian Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            if (Mean.IsPointMass)
                return SampleConditional(Mean.Point, Precision, result);
            if (result == default(VectorGaussian))
                result = new VectorGaussian(Mean.Dimension);
            // R = Prec/(Prec + Mean.Prec)
            PositiveDefiniteMatrix R = Precision + Mean.Precision;
            R.SetToProduct(Precision, R.Inverse());
            for (int i = 0; i < Mean.Dimension; i++)
            {
                if (double.IsPositiveInfinity(Mean.Precision[i, i]))
                    R[i, i] = 1;
            }
            result.Precision.SetToProduct(R, Mean.Precision);
            result.Precision.Symmetrize();
            for (int i = 0; i < Mean.Dimension; i++)
            {
                if (double.IsPositiveInfinity(Mean.Precision[i, i]))
                {
                    for (int j = 0; j < Mean.Dimension; j++)
                    {
                        result.Precision[i, j] = 0;
                        result.Precision[j, i] = 0;
                    }
                    result.Precision[i, i] = 1;
                }
            }
            result.MeanTimesPrecision.SetToProduct(R, Mean.MeanTimesPrecision);
            return result;
        }

        /// <summary />
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian SampleAverageConditionalInit(Vector Mean)
        {
            return VectorGaussian.Uniform(Mean.Count);
        }

        /// <summary />
        /// <param name="Mean">Incoming message from <c>mean</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian SampleAverageConditionalInit([IgnoreDependency] VectorGaussian Mean)
        {
            return new VectorGaussian(Mean.Dimension);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian MeanAverageConditional(Vector Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Sample, Precision, result);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>mean</c> as the random arguments are varied. The formula is <c>proj[p(mean) sum_(sample) p(sample) factor(sample,mean,precision)]/p(mean)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        public static VectorGaussian MeanAverageConditional([SkipIfUniform] VectorGaussian Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleAverageConditional(Sample, Precision, result);
        }

        /// <summary>EP message to <c>precision</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>precision</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart PrecisionAverageConditional(Vector Sample, Vector Mean, Wishart result)
        {
            return PrecisionConditional(Sample, Mean, result);
        }

        //-- VMP ----------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precisionMean">Buffer <c>precisionMean</c>.</param>
        /// <param name="precisionMeanLogDet">Buffer <c>precisionMeanLogDet</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,mean,precision) p(sample,mean,precision) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            [Proper] VectorGaussian mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            [Proper] Wishart precision,
            PositiveDefiniteMatrix precisionMean,
            double precisionMeanLogDet)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, MeanMean, MeanVariance, precision, precisionMean, precisionMeanLogDet);
            if (mean.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean.Point, precision, precisionMean, precisionMeanLogDet);
            if (precision.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean, MeanMean, MeanVariance, precision.Point);

            return ComputeAverageLogFactor(SampleMean, SampleVariance, MeanMean, MeanVariance, precisionMeanLogDet, precisionMean);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precisionMean">Buffer <c>precisionMean</c>.</param>
        /// <param name="precisionMeanLogDet">Buffer <c>precisionMeanLogDet</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(precision) p(precision) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            Vector sample, Vector mean, [Proper] Wishart precision, PositiveDefiniteMatrix precisionMean, double precisionMeanLogDet)
        {
            if (precision.IsPointMass)
                return AverageLogFactor(sample, mean, precision.Point);
            else
                return ComputeAverageLogFactor(sample, mean, precisionMeanLogDet, precisionMean);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Vector sample, Vector mean, PositiveDefiniteMatrix precision)
        {
            return ComputeAverageLogFactor(sample, mean, precision.LogDeterminant(ignoreInfinity: true), precision);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector mean,
            PositiveDefiniteMatrix precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision);
            else
                return ComputeAverageLogFactor(SampleMean, SampleVariance, mean, precision.LogDeterminant(ignoreInfinity: true), precision);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(mean) p(mean) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            Vector sample, [Proper] VectorGaussian mean, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, PositiveDefiniteMatrix precision)
        {
            return AverageLogFactor(mean, MeanMean, MeanVariance, sample, precision);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precisionMean">Buffer <c>precisionMean</c>.</param>
        /// <param name="precisionMeanLogDet">Buffer <c>precisionMeanLogDet</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(mean,precision) p(mean,precision) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            Vector sample,
            [Proper] VectorGaussian mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            [Proper] Wishart precision,
            PositiveDefiniteMatrix precisionMean,
            double precisionMeanLogDet)
        {
            return AverageLogFactor(mean, MeanMean, MeanVariance, sample, precision, precisionMean, precisionMeanLogDet);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precisionMean">Buffer <c>precisionMean</c>.</param>
        /// <param name="precisionMeanLogDet">Buffer <c>precisionMeanLogDet</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,precision) p(sample,precision) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector mean,
            [Proper] Wishart precision,
            PositiveDefiniteMatrix precisionMean,
            double precisionMeanLogDet)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, precision, precisionMean, precisionMeanLogDet);
            if (precision.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean, precision.Point);

            return ComputeAverageLogFactor(SampleMean, SampleVariance, mean, precisionMeanLogDet, precisionMean);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="precision">Constant value for <c>precision</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,mean) p(sample,mean) log(factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static double AverageLogFactor(
            [Proper] VectorGaussian sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            [Proper] VectorGaussian mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            PositiveDefiniteMatrix precision)
        {
            if (sample.IsPointMass)
                return AverageLogFactor(sample.Point, mean, MeanMean, MeanVariance, precision);
            if (mean.IsPointMass)
                return AverageLogFactor(sample, SampleMean, SampleVariance, mean.Point, precision);

            return ComputeAverageLogFactor(SampleMean, SampleVariance, MeanMean, MeanVariance, precision.LogDeterminant(ignoreInfinity: true), precision);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="SampleMean">Mean of incoming message from 'sample'</param>
        /// <param name="SampleVariance">Variance of incoming message from 'sample'</param>
        /// <param name="MeanMean">Mean of incoming message from 'mean'</param>
        /// <param name="MeanVariance">Variance of incoming message from 'mean'</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            double precision_Elogx,
            PositiveDefiniteMatrix precision_Ex)
        {
            int dim = SampleMean.Count;
            int nonzeroDims = 0;
            double precTimesVariance = 0.0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                if (double.IsPositiveInfinity(precision_Ex[i, i]))
                {
                    if (SampleMean[i] != MeanMean[i] || SampleVariance[i, i] + MeanVariance[i, i] > 0)
                        return double.NegativeInfinity;
                }
                else
                {
                    nonzeroDims++;
                    double sum = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += precision_Ex[i, j] * (SampleMean[j] - MeanMean[j]);
                        precTimesVariance += precision_Ex[i, j] * (SampleVariance[i, j] + MeanVariance[i, j]);
                    }
                    precTimesDiff += sum * (SampleMean[i] - MeanMean[i]);
                }
            }
            return -nonzeroDims * MMath.LnSqrt2PI + 0.5 * (precision_Elogx - precTimesVariance - precTimesDiff);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="SampleMean">Mean of incoming sample message</param>
        /// <param name="SampleVariance">Variance of incoming sample message</param>
        /// <param name="mean">Constant value for 'mean'.</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector mean,
            double precision_Elogx,
            PositiveDefiniteMatrix precision_Ex)
        {
            int dim = mean.Count;
            int nonzeroDims = 0;
            double precTimesVariance = 0.0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                if (double.IsPositiveInfinity(precision_Ex[i, i]))
                {
                    if (SampleMean[i] != mean[i] || SampleVariance[i, i] > 0)
                        return double.NegativeInfinity;
                }
                else
                {
                    nonzeroDims++;
                    double sum = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += precision_Ex[i, j] * (SampleMean[j] - mean[j]);
                        precTimesVariance += precision_Ex[i, j] * SampleVariance[j, i];
                    }
                    precTimesDiff += sum * (SampleMean[i] - mean[i]);
                }
            }
            return -nonzeroDims * MMath.LnSqrt2PI + 0.5 * (precision_Elogx - precTimesVariance - precTimesDiff);
        }

        /// <summary>
        /// Helper method for computing average log factor
        /// </summary>
        /// <param name="sample">Constant value for 'sample'.</param>
        /// <param name="mean">Constant value for 'mean'.</param>
        /// <param name="precision_Elogx">Expected log value of the incoming message from 'precision'</param>
        /// <param name="precision_Ex">Expected value of incoming message from 'precision'</param>
        /// <returns>Computed average log factor</returns>
        private static double ComputeAverageLogFactor(Vector sample, Vector mean, double precision_Elogx, PositiveDefiniteMatrix precision_Ex)
        {
            int dim = mean.Count;
            int nonzeroDims = 0;
            double precTimesDiff = 0.0;
            for (int i = 0; i < dim; i++)
            {
                if (double.IsPositiveInfinity(precision_Ex[i, i]))
                {
                    if (sample[i] != mean[i])
                        return double.NegativeInfinity;
                }
                else
                {
                    nonzeroDims++;
                    double sum = 0.0;
                    for (int j = 0; j < dim; j++)
                    {
                        sum += precision_Ex[i, j] * (sample[j] - mean[j]);
                    }
                    precTimesDiff += sum * (sample[i] - mean[i]);
                }
            }
            return -nonzeroDims * MMath.LnSqrt2PI + 0.5 * (precision_Elogx - precTimesDiff);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SampleAverageLogarithm(Vector Mean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Mean, Precision, result);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="Mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="Precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="PrecisionMean">Buffer <c>PrecisionMean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(mean,precision) p(mean,precision) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Precision" /> is not a proper distribution.</exception>
        public static VectorGaussian SampleAverageLogarithm(
            [Proper] VectorGaussian Mean,
            Vector MeanMean,
            [Proper] Wishart Precision,
            PositiveDefiniteMatrix PrecisionMean,
            VectorGaussian result)
        {
            return SampleAverageLogarithm(MeanMean, PrecisionMean, result);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(mean) p(mean) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        public static VectorGaussian SampleAverageLogarithm([Proper] VectorGaussian mean, Vector MeanMean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            if (result == default(VectorGaussian))
                result = new VectorGaussian(MeanMean.Count);
            result.Precision.SetTo(Precision);
            result.MeanTimesPrecision.SetToProduct(result.Precision, MeanMean);
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="Precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="PrecisionMean">Buffer <c>PrecisionMean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(precision) p(precision) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Precision" /> is not a proper distribution.</exception>
        public static VectorGaussian SampleAverageLogarithm(Vector Mean, [Proper] Wishart Precision, PositiveDefiniteMatrix PrecisionMean, VectorGaussian result)
        {
            return SampleAverageLogarithm(Mean, PrecisionMean, result);
        }

        /// <summary />
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian SampleAverageLogarithmInit(Vector Mean)
        {
            return VectorGaussian.Uniform(Mean.Count);
        }

        /// <summary />
        /// <param name="Mean">Incoming message from <c>mean</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian SampleAverageLogarithmInit([IgnoreDependency] VectorGaussian Mean)
        {
            return new VectorGaussian(Mean.Dimension);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian MeanAverageLogarithm(Vector Sample, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleConditional(Sample, Precision, result);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="Precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="PrecisionMean">Buffer <c>PrecisionMean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>mean</c>. The formula is <c>exp(sum_(sample,precision) p(sample,precision) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Precision" /> is not a proper distribution.</exception>
        public static VectorGaussian MeanAverageLogarithm(
            [Proper] VectorGaussian Sample, Vector SampleMean, [Proper] Wishart Precision, PositiveDefiniteMatrix PrecisionMean, VectorGaussian result)
        {
            return SampleAverageLogarithm(Sample, SampleMean, Precision, PrecisionMean, result);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="Precision">Constant value for <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>mean</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        public static VectorGaussian MeanAverageLogarithm(
            [Proper] VectorGaussian Sample, Vector SampleMean, PositiveDefiniteMatrix Precision, VectorGaussian result)
        {
            return SampleAverageLogarithm(Sample, SampleMean, Precision, result);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="PrecisionMean">Buffer <c>PrecisionMean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>mean</c>. The formula is <c>exp(sum_(precision) p(precision) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Precision" /> is not a proper distribution.</exception>
        public static VectorGaussian MeanAverageLogarithm(Vector Sample, [Proper] Wishart Precision, PositiveDefiniteMatrix PrecisionMean, VectorGaussian result)
        {
            return SampleAverageLogarithm(Sample, Precision, PrecisionMean, result);
        }

        /// <summary>VMP message to <c>precision</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>precision</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart PrecisionAverageLogarithm(Vector Sample, Vector Mean, Wishart result)
        {
            return PrecisionConditional(Sample, Mean, result);
        }

        /// <summary>VMP message to <c>precision</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="Mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>precision</c>. The formula is <c>exp(sum_(sample,mean) p(sample,mean) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Mean" /> is not a proper distribution.</exception>
        public static Wishart PrecisionAverageLogarithm(
            [Proper] VectorGaussian Sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            [Proper] VectorGaussian Mean,
            Vector MeanMean,
            PositiveDefiniteMatrix MeanVariance,
            Wishart result)
        {
            if (Sample.IsPointMass)
                return PrecisionAverageLogarithm(Sample.Point, Mean, MeanMean, MeanVariance, result);
            if (Mean.IsPointMass)
                return PrecisionAverageLogarithm(Sample, SampleMean, SampleVariance, Mean.Point, result);
            // The formula is exp(int_x int_mean p(x) p(mean) log N(x;mean,1/prec)) =
            // exp(-0.5 prec E[(x-mean)^2] + 0.5 log(prec)) =
            // Gamma(prec; 0.5, 0.5*E[(x-mean)^2])
            // E[(x-mean)^2] = E[x^2] - 2 E[x] E[mean] + E[mean^2] = var(x) + (E[x]-E[mean])^2 + var(mean)
            if (result == default(Wishart))
                result = new Wishart(Sample.Dimension);
            // we want shape - (d+1)/2 = 0.5, therefore shape = (d+2)/2
            result.Shape = 0.5 * (result.Dimension + 2);
            Vector diff = SampleMean - MeanMean;
            result.Rate.SetToOuter(diff, diff);
            result.Rate.SetToSum(result.Rate, SampleVariance);
            result.Rate.SetToSum(result.Rate, MeanVariance);
            result.Rate.Scale(0.5);
            return result;
        }

        /// <summary>VMP message to <c>precision</c>.</summary>
        /// <param name="Sample">Constant value for <c>sample</c>.</param>
        /// <param name="Mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="MeanMean">Buffer <c>MeanMean</c>.</param>
        /// <param name="MeanVariance">Buffer <c>MeanVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>precision</c>. The formula is <c>exp(sum_(mean) p(mean) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Mean" /> is not a proper distribution.</exception>
        public static Wishart PrecisionAverageLogarithm(
            Vector Sample, [Proper] VectorGaussian Mean, Vector MeanMean, PositiveDefiniteMatrix MeanVariance, Wishart result)
        {
            if (Mean.IsPointMass)
                return PrecisionAverageLogarithm(Sample, Mean.Point, result);
            // The formula is exp(int_x int_mean p(x) p(mean) log N(x;mean,1/prec)) =
            // exp(-0.5 prec E[(x-mean)^2] + 0.5 log(prec)) =
            // Gamma(prec; 0.5, 0.5*E[(x-mean)^2])
            // E[(x-mean)^2] = E[x^2] - 2 E[x] E[mean] + E[mean^2] = var(x) + (E[x]-E[mean])^2 + var(mean)
            if (result == default(Wishart))
                result = new Wishart(Sample.Count);
            result.Shape = 0.5 * (result.Dimension + 2);
            Vector diff = Sample - MeanMean;
            result.Rate.SetToOuter(diff, diff);
            result.Rate.SetToSum(result.Rate, MeanVariance);
            result.Rate.Scale(0.5);
            return result;
        }

        /// <summary>VMP message to <c>precision</c>.</summary>
        /// <param name="Sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="SampleMean">Buffer <c>SampleMean</c>.</param>
        /// <param name="SampleVariance">Buffer <c>SampleVariance</c>.</param>
        /// <param name="Mean">Constant value for <c>mean</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>precision</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,mean,precision)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Sample" /> is not a proper distribution.</exception>
        public static Wishart PrecisionAverageLogarithm(
            [Proper] VectorGaussian Sample,
            Vector SampleMean,
            PositiveDefiniteMatrix SampleVariance,
            Vector Mean,
            Wishart result)
        {
            return PrecisionAverageLogarithm(Mean, Sample, SampleMean, SampleVariance, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="VectorGaussian.SampleFromMeanAndVariance(Vector, PositiveDefiniteMatrix)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(VectorGaussian), "SampleFromMeanAndVariance")]
    [Quality(QualityBand.Stable)]
    public static class VectorGaussianFromMeanAndVarianceOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,variance))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            VectorGaussian to_sample = SampleAverageConditional(mean, variance);
            return to_sample.GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector sample, Vector mean, PositiveDefiniteMatrix variance)
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
        public static double AverageLogFactor(Vector sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            return LogAverageFactor(sample, mean, variance);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(VectorGaussian sample, [Fresh] VectorGaussian to_sample)
        {
            return LogAverageFactor(sample, to_sample);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="to_mean">Outgoing message to <c>mean</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(mean) p(mean) log(factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Vector sample, VectorGaussian mean, [Fresh] VectorGaussian to_mean)
        {
            return AverageLogFactor(mean, to_mean);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,mean,variance))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(VectorGaussian sample, [Fresh] VectorGaussian to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(mean) p(mean) factor(sample,mean,variance))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector sample, VectorGaussian mean, PositiveDefiniteMatrix variance)
        {
            return SampleAverageConditional(sample, variance).GetLogAverageOf(mean);
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
        public static double LogEvidenceRatio(VectorGaussian sample, Vector mean, PositiveDefiniteMatrix variance)
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SampleAverageLogarithm(Vector mean, PositiveDefiniteMatrix variance)
        {
            return VectorGaussian.FromMeanAndVariance(mean, variance);
        }

        /// <summary>VMP message to <c>mean</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>The outgoing VMP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian MeanAverageLogarithm(Vector sample, PositiveDefiniteMatrix variance)
        {
            return SampleAverageLogarithm(sample, variance);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="mean">Constant value for <c>mean</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian SampleAverageConditional(Vector mean, PositiveDefiniteMatrix variance)
        {
            return VectorGaussian.FromMeanAndVariance(mean, variance);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="variance">Constant value for <c>variance</c>.</param>
        /// <returns>The outgoing EP message to the <c>mean</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>mean</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian MeanAverageConditional(Vector sample, PositiveDefiniteMatrix variance)
        {
            return SampleAverageConditional(sample, variance);
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="VectorGaussian.Sample(Vector, PositiveDefiniteMatrix)" /></description></item><item><description><see cref="Factor.VectorGaussian(Vector, PositiveDefiniteMatrix)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(VectorGaussian), "Sample", typeof(Vector), typeof(PositiveDefiniteMatrix))]
    [FactorMethod(new string[] { "sample", "mean", "precision" }, typeof(Factor), "VectorGaussian")]
    [Buffers("SampleMean", "SampleVariance", "MeanMean", "MeanVariance", "PrecisionMean", "PrecisionMeanLogDet")]
    [Quality(QualityBand.Experimental)]
    public static class VectorGaussianOp_Laplace2
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(mean,precision) p(mean,precision) factor(sample,mean,precision))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(Vector sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision)
        {
            // int_(x,m) f(x,m,r) p(x) p(m) dx dm = N(mx;mm, vx+vm+1/r) = |vx+vm+1/r|^(-1/2) exp(-0.5 tr((vx+vm+1/r)^(-1) (mx-mm)(mx-mm)'))
            int dim = precision.Dimension;
            Wishart rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            PositiveDefiniteMatrix ir = r.Inverse();
            Vector mm = Vector.Zero(dim);
            PositiveDefiniteMatrix vm = new PositiveDefiniteMatrix(dim, dim);
            mean.GetMeanAndVariance(mm, vm);
            PositiveDefiniteMatrix v = vm;
            vm = null;
            v.SetToSum(v, ir);
            PositiveDefiniteMatrix iv = v.Inverse();
            Matrix ivr = ir * iv;
            Vector m = mm;
            mm = null;
            m.SetToDifference(sample, m);
            double result = -dim * MMath.LnSqrt2PI + 0.5 * iv.LogDeterminant() - 0.5 * iv.QuadraticForm(m);
            result += precision.GetLogProb(r) - rPost.GetLogProb(r);
            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(mean,precision) p(mean,precision) factor(sample,mean,precision))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(Vector sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision)
        {
            return LogAverageFactor(sample, mean, precision, to_precision);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,mean,precision) p(sample,mean,precision) factor(sample,mean,precision))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogAverageFactor(VectorGaussian sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision)
        {
            // int_(x,m) f(x,m,r) p(x) p(m) dx dm = N(mx;mm, vx+vm+1/r) = |vx+vm+1/r|^(-1/2) exp(-0.5 tr((vx+vm+1/r)^(-1) (mx-mm)(mx-mm)'))
            int dim = precision.Dimension;
            Wishart rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            PositiveDefiniteMatrix ir = r.Inverse();
            Vector mx = Vector.Zero(dim);
            PositiveDefiniteMatrix vx = new PositiveDefiniteMatrix(dim, dim);
            sample.GetMeanAndVariance(mx, vx);
            Vector mm = Vector.Zero(dim);
            PositiveDefiniteMatrix vm = new PositiveDefiniteMatrix(dim, dim);
            mean.GetMeanAndVariance(mm, vm);
            PositiveDefiniteMatrix v = vx;
            vx = null;
            v.SetToSum(v, vm);
            v.SetToSum(v, ir);
            PositiveDefiniteMatrix iv = v.Inverse();
            Matrix ivr = ir * iv;
            Vector m = mx;
            mx = null;
            m.SetToDifference(m, mm);
            double result = -dim * MMath.LnSqrt2PI + 0.5 * iv.LogDeterminant() - 0.5 * iv.QuadraticForm(m);
            result += precision.GetLogProb(r) - rPost.GetLogProb(r);
            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="to_sample">Previous outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,mean,precision) p(sample,mean,precision) factor(sample,mean,precision) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(VectorGaussian sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision, VectorGaussian to_sample)
        {
            return LogAverageFactor(sample, mean, precision, to_precision) - to_sample.GetLogAverageOf(sample);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(mean,precision) p(mean,precision) factor(sample,mean,precision)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static VectorGaussian SampleAverageConditional(VectorGaussian sample, [SkipIfUniform] VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision, VectorGaussian result)
        {
            var rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            return VectorGaussianOp.SampleAverageConditional(mean, r, result);
        }

        /// <summary>EP message to <c>mean</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>mean</c>.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>mean</c> as the random arguments are varied. The formula is <c>proj[p(mean) sum_(sample,precision) p(sample,precision) factor(sample,mean,precision)]/p(mean)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static VectorGaussian MeanAverageConditional([SkipIfUniform] VectorGaussian sample, VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision, VectorGaussian result)
        {
            return SampleAverageConditional(mean, sample, precision, to_precision, result);
        }

        /// <summary>EP message to <c>precision</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="mean">Incoming message from <c>mean</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="precision">Incoming message from <c>precision</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_precision">Previous outgoing message to <c>precision</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>precision</c> as the random arguments are varied. The formula is <c>proj[p(precision) sum_(sample,mean) p(sample,mean) factor(sample,mean,precision)]/p(precision)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="mean" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="precision" /> is not a proper distribution.</exception>
        public static Wishart PrecisionAverageConditional([SkipIfUniform] VectorGaussian sample, [SkipIfUniform] VectorGaussian mean, [Proper] Wishart precision, Wishart to_precision, Wishart result)
        {
            // int_(x,m) f(x,m,r) p(x) p(m) dx dm = N(mx;mm, vx+vm+1/r) = |vx+vm+1/r|^(-1/2) exp(-0.5 tr((vx+vm+1/r)^(-1) (mx-mm)(mx-mm)'))
            // log f(r) = -0.5 log|v+1/r| -0.5 tr((v+1/r)^(-1) S)
            // dlogf(r) = 0.5 tr((1/r) (v+1/r)^(-1) (1/r) dr) - 0.5 tr((1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) dr)
            // tr(r dlogf') = 0.5 tr(r (d(1/r) (v+1/r)^(-1) (1/r) + (1/r) d(v+1/r)^(-1) (1/r) + (1/r) (v+1/r)^(-1) d(1/r)))
            //               -0.5 tr(r (d(1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) ...
            //              = -tr((1/r) (v+1/r)^(-1) (1/r) dr) + 0.5 tr((1/r) (v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r) dr)
            //                +tr((1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) dr) - tr((1/r) (v+1/r)^(-1) S (v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r) dr)
            // tr(r tr(r dlogf')/dr) = -tr((v+1/r)^(-1) (1/r)) + 0.5 tr((v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r))
            //                         +tr((v+1/r)^(-1) S (v+1/r)^(-1) (1/r)) - tr((v+1/r)^(-1) S (v+1/r)^(-1) (1/r) (v+1/r)^(-1) (1/r))
            int dim = sample.Dimension;
            Wishart rPost = precision * to_precision;
            PositiveDefiniteMatrix r = rPost.GetMean();
            PositiveDefiniteMatrix ir = r.Inverse();
            Vector mx = Vector.Zero(dim);
            PositiveDefiniteMatrix vx = new PositiveDefiniteMatrix(dim, dim);
            sample.GetMeanAndVariance(mx, vx);
            Vector mm = Vector.Zero(dim);
            PositiveDefiniteMatrix vm = new PositiveDefiniteMatrix(dim, dim);
            mean.GetMeanAndVariance(mm, vm);
            PositiveDefiniteMatrix v = vx;
            vx = null;
            v.SetToSum(v, vm);
            v.SetToSum(v, ir);
            PositiveDefiniteMatrix iv = v.Inverse();
            Matrix ivr = ir * iv;
            Vector m = mx;
            mx = null;
            m.SetToDifference(m, mm);
            Vector ivrm = mm;
            mm = null;
            ivrm.SetToProduct(ivr, m);
            PositiveDefiniteMatrix ivrs = vm;
            vm = null;
            ivrs.SetToOuter(ivrm, ivrm);
            var ivrr = new PositiveDefiniteMatrix(dim, dim);
            ivrr.SetToProduct(ivr, ir);
            ivrr.Symmetrize();
            double xxddlogf = -ivr.Trace() + 0.5 * Matrix.TraceOfProduct(ivrr, iv) + Matrix.TraceOfProduct(ivrs, r) - Matrix.TraceOfProduct(ivrs, iv);
            // result.Rate holds dlogf
            result.Rate.SetToSum(0.5, ivrr, -0.5, ivrs);
            LowerTriangularMatrix rChol = new LowerTriangularMatrix(dim, dim);
            rChol.SetToCholesky(r);
            result.SetDerivatives(rChol, ir, result.Rate, xxddlogf, GaussianOp.ForceProper);
            return result;
        }
    }
}
