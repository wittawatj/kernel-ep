// (C) Copyright 2009-2010 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Linq;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Wishart.SampleFromShapeAndScale(double, PositiveDefiniteMatrix)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Wishart), "SampleFromShapeAndScale")]
    [Quality(QualityBand.Stable)]
    public static class WishartFromShapeAndScaleOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="scale">Constant value for <c>scale</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,shape,scale))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix scale)
        {
            Wishart to_sample = SampleAverageConditional(shape, scale);
            return to_sample.GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="scale">Constant value for <c>scale</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,shape,scale))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix scale)
        {
            return LogAverageFactor(sample, shape, scale);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="scale">Constant value for <c>scale</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,shape,scale) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Wishart sample, double shape, PositiveDefiniteMatrix scale)
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="scale">Constant value for <c>scale</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,shape,scale))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix scale)
        {
            return LogAverageFactor(sample, shape, scale);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample) p(sample) log(factor(sample,shape,scale))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Wishart sample, [Fresh] Wishart to_sample)
        {
            return LogAverageFactor(sample, to_sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,shape,scale))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Wishart sample, [Fresh] Wishart to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="scale">Constant value for <c>scale</c>.</param>
        /// <returns>The outgoing VMP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart SampleAverageLogarithm(double shape, PositiveDefiniteMatrix scale)
        {
            return Wishart.FromShapeAndScale(shape, scale);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="scale">Constant value for <c>scale</c>.</param>
        /// <returns>The outgoing EP message to the <c>sample</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart SampleAverageConditional(double shape, PositiveDefiniteMatrix scale)
        {
            return Wishart.FromShapeAndScale(shape, scale);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Wishart.SampleFromShapeAndRate(double, PositiveDefiniteMatrix)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Wishart), "SampleFromShapeAndRate", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Quality(QualityBand.Stable)]
    public static class WishartFromShapeAndRateOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Constant value for <c>rate</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,shape,rate))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix rate)
        {
            int dimension = sample.Rows;
            Wishart to_sample = SampleAverageConditional(shape, rate, new Wishart(dimension));
            return to_sample.GetLogProb(sample);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Constant value for <c>rate</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,shape,rate))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(rate) p(rate) factor(sample,shape,rate))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(PositiveDefiniteMatrix sample, double shape, Wishart rate)
        {
            // f(X,a,B) = |X|^(a-c) |B|^a/Gamma_d(a) exp(-tr(BX)) = |X|^(-2c) Gamma_d(a+c)/Gamma_d(a) W(B; a+c, X)
            // p(B) = |B|^(a'-c) |B'|^(a')/Gamma_d(a') exp(-tr(BB'))
            // int_B f p(B) dB = |X|^(a-c) Gamma_d(a+a')/Gamma_d(a)/Gamma_d(a') |B'|^(a') |B'+X|^(-(a+a'))
            int dimension = sample.Rows;
            double c = 0.5 * (dimension + 1);
            Wishart to_rate = new Wishart(dimension);
            to_rate = RateAverageConditional(sample, shape, to_rate);
            return rate.GetLogAverageOf(to_rate) - 2 * c * sample.LogDeterminant() + MMath.GammaLn(shape + c, dimension) - MMath.GammaLn(shape, dimension);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(rate) p(rate) factor(sample,shape,rate))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sample, double shape, Wishart rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Constant value for <c>rate</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,shape,rate) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Wishart sample, double shape, PositiveDefiniteMatrix rate)
        {
            return 0.0;
        }

        /// <summary>EP message to <c>rate</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>rate</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart RateAverageConditional(PositiveDefiniteMatrix sample, double shape, Wishart result)
        {
            int dimension = result.Dimension;
            result.Shape = shape + 0.5 * (dimension + 1);
            result.Rate.SetTo(sample);
            return result;
        }

        // VMP //////////////////////////////////////////////////////////////////////////////////////////////////////////

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Constant value for <c>rate</c>.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sample,shape,rate))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(PositiveDefiniteMatrix sample, double shape, PositiveDefiniteMatrix rate)
        {
            return LogAverageFactor(sample, shape, rate);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample) p(sample) factor(sample,shape,rate))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Wishart sample, [Fresh] Wishart to_sample)
        {
            return to_sample.GetLogAverageOf(sample);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>Average of the factor's log-value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>sum_(sample,rate) p(sample,rate) log(factor(sample,shape,rate))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rate" /> is not a proper distribution.</exception>
        public static double AverageLogFactor([Proper] Wishart sample, double shape, [Proper] Wishart rate)
        {
            // factor = (a-(d+1)/2)*logdet(X) -tr(X*B) + a*logdet(B) - GammaLn(a,d)
            int dimension = sample.Dimension;
            return (shape - (dimension + 1) * 0.5) * sample.GetMeanLogDeterminant() - Matrix.TraceOfProduct(sample.GetMean(), rate.GetMean()) + shape * rate.GetMeanLogDeterminant() -
                         MMath.GammaLn(shape, dimension);
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Constant value for <c>rate</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart SampleAverageLogarithm(double shape, PositiveDefiniteMatrix rate, Wishart result)
        {
            result.Shape = shape;
            result.Rate.SetTo(rate);
            return result;
        }

        /// <summary>VMP message to <c>sample</c>.</summary>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>sample</c>. The formula is <c>exp(sum_(rate) p(rate) log(factor(sample,shape,rate)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rate" /> is not a proper distribution.</exception>
        public static Wishart SampleAverageLogarithm(double shape, [SkipIfUniform] Wishart rate, Wishart result)
        {
            result.Shape = shape;
            rate.GetMean(result.Rate);
            return result;
        }

        /// <summary>VMP message to <c>rate</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>rate</c>. The formula is <c>exp(sum_(sample) p(sample) log(factor(sample,shape,rate)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static Wishart RateAverageLogarithm([SkipIfUniform] Wishart sample, double shape, Wishart result)
        {
            int dimension = result.Dimension;
            result.Shape = shape + 0.5 * (dimension + 1);
            sample.GetMean(result.Rate);
            return result;
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Constant value for <c>rate</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>sample</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart SampleAverageConditional(double shape, PositiveDefiniteMatrix rate, Wishart result)
        {
            return SampleAverageLogarithm(shape, rate, result);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Wishart.SampleFromShapeAndRate(double, PositiveDefiniteMatrix)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Wishart), "SampleFromShapeAndRate", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Quality(QualityBand.Experimental)]
    public static class WishartFromShapeAndRateOp_Laplace2
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,rate) p(sample,rate) factor(sample,shape,rate))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Wishart sample, double shape, Wishart rate, [Fresh] Wishart to_sample)
        {
            // int_R f(Y,R) p(R) dR = |Y|^(a-c) |Y+B_r|^-(a+a_r) Gamma_d(a+a_r)/Gamma_d(a)/Gamma_d(a_r) |B_r|^a_r
            int dim = sample.Dimension;
            double c = 0.5 * (dim + 1);
            double shape2 = shape + rate.Shape;
            Wishart samplePost = sample * to_sample;
            PositiveDefiniteMatrix y = samplePost.GetMean();
            PositiveDefiniteMatrix yPlusBr = y + rate.Rate;
            double result = (shape - c) * y.LogDeterminant() - shape2 * yPlusBr.LogDeterminant() + sample.GetLogProb(y) - samplePost.GetLogProb(y);
            result += MMath.GammaLn(shape2, dim) - MMath.GammaLn(shape, dim) - MMath.GammaLn(rate.Shape, dim);
            result += rate.Shape * rate.Rate.LogDeterminant();
            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>.</param>
        /// <param name="to_sample">Outgoing message to <c>sample</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sample,rate) p(sample,rate) factor(sample,shape,rate) / sum_sample p(sample) messageTo(sample))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Wishart sample, double shape, Wishart rate, [Fresh] Wishart to_sample)
        {
            return LogAverageFactor(sample, shape, rate, to_sample) - to_sample.GetLogAverageOf(sample);
        }

        /// <summary>EP message to <c>rate</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>.</param>
        /// <param name="to_rate">Previous outgoing message to <c>rate</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>rate</c> as the random arguments are varied. The formula is <c>proj[p(rate) sum_(sample) p(sample) factor(sample,shape,rate)]/p(rate)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sample" /> is not a proper distribution.</exception>
        public static Wishart RateAverageConditional([SkipIfUniform] Wishart sample, double shape, Wishart rate, Wishart to_rate, Wishart result)
        {
            // f(Y,R) = |Y|^(a-c) |R|^a exp(-tr(YR))
            // p(Y) = |Y|^(a_y-c) exp(-tr(YB_y)
            // p(R) = |R|^(a_r-c) exp(-tr(RB_r)
            // int_Y f(Y,R) p(Y) dY = |R|^a |R+B_y|^-(a+a_y-c)
            int dim = sample.Dimension;
            double c = 0.5 * (dim + 1);
            double shape2 = shape - c + sample.Shape;
            Wishart ratePost = rate * to_rate;
            PositiveDefiniteMatrix r = ratePost.GetMean();
            PositiveDefiniteMatrix rby = r + sample.Rate;
            PositiveDefiniteMatrix invrby = rby.Inverse();
            PositiveDefiniteMatrix rInvrby = rby;
            rInvrby.SetToProduct(r, invrby);
            double xxddlogp = Matrix.TraceOfProduct(rInvrby, rInvrby) * shape2;
            double delta = -xxddlogp / dim;
            PositiveDefiniteMatrix invR = r.Inverse();
            PositiveDefiniteMatrix dlogp = invrby;
            dlogp.Scale(-shape2);
            LowerTriangularMatrix rChol = new LowerTriangularMatrix(dim, dim);
            rChol.SetToCholesky(r);
            result.SetDerivatives(rChol, invR, dlogp, xxddlogp, GammaFromShapeAndRateOp.ForceProper, shape);
            return result;
        }

        public static Wishart SampleAverageConditional2(double shape, [Proper] Wishart rate, Wishart to_rate, Wishart result)
        {
            Wishart ratePost = rate * to_rate;
            PositiveDefiniteMatrix r = ratePost.GetMean();
            return WishartFromShapeAndRateOp.SampleAverageConditional(shape, r, result);
        }

        /// <summary>EP message to <c>sample</c>.</summary>
        /// <param name="sample">Incoming message from <c>sample</c>.</param>
        /// <param name="shape">Constant value for <c>shape</c>.</param>
        /// <param name="rate">Incoming message from <c>rate</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_rate">Previous outgoing message to <c>rate</c>.</param>
        /// <param name="to_sample">Previous outgoing message to <c>sample</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sample</c> as the random arguments are varied. The formula is <c>proj[p(sample) sum_(rate) p(rate) factor(sample,shape,rate)]/p(sample)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="rate" /> is not a proper distribution.</exception>
        public static Wishart SampleAverageConditional(Wishart sample, double shape, [Proper] Wishart rate, Wishart to_rate, Wishart to_sample, Wishart result)
        {
            if (sample.IsUniform())
                return SampleAverageConditional2(shape, rate, to_rate, result);
            // f(Y,R) = |Y|^(a-c) |R|^a exp(-tr(YR))
            // p(Y) = |Y|^(a_y-c) exp(-tr(YB_y)
            // p(R) = |R|^(a_r-c) exp(-tr(RB_r)
            // int_R f(Y,R) p(R) dR = |Y|^(a-c) |Y+B_r|^-(a+a_r)
            int dim = sample.Dimension;
            double c = 0.5 * (dim + 1);
            double shape2 = shape + rate.Shape;
            Wishart samplePost = sample * to_sample;
            PositiveDefiniteMatrix y = samplePost.GetMean();
            PositiveDefiniteMatrix yPlusBr = y + rate.Rate;
            PositiveDefiniteMatrix invyPlusBr = yPlusBr.Inverse();
            PositiveDefiniteMatrix yInvyPlusBr = yPlusBr;
            yInvyPlusBr.SetToProduct(y, invyPlusBr);
            double xxddlogf = shape2 * Matrix.TraceOfProduct(yInvyPlusBr, yInvyPlusBr);
            PositiveDefiniteMatrix invY = y.Inverse();
            //double delta = -xxddlogf / dim;
            //result.Shape = delta + shape;
            //result.Rate.SetToSum(delta, invY, shape2, invyPlusBr);
            LowerTriangularMatrix yChol = new LowerTriangularMatrix(dim, dim);
            yChol.SetToCholesky(y);
            PositiveDefiniteMatrix dlogp = invyPlusBr;
            dlogp.Scale(-shape2);
            result.SetDerivatives(yChol, invY, dlogp, xxddlogf, GammaFromShapeAndRateOp.ForceProper, shape - c);
            if (result.Rate.Any(x => double.IsNaN(x)))
                throw new Exception("result.Rate is nan");
            return result;
        }
    }
}
