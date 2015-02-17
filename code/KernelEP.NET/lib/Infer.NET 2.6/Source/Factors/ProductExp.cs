// (C) Copyright 2009-2010 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;

    /// <summary>Provides outgoing messages for <see cref="Factor.ProductExp(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "ProductExp", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class ProductExpOp
    {
        /// <summary>VMP message to <c>productExp</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>productExp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>productExp</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(productExp,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductExpAverageLogarithm([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            double ma, mb, va, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            if (Double.IsPositiveInfinity(va) || Double.IsPositiveInfinity(vb))
                return Gaussian.Uniform();
            double mp = ma * Math.Exp(mb + vb / 2.0);
            return Gaussian.FromMeanAndVariance(mp, (va + ma * ma) * Math.Exp(2.0 * (mb + vb)) - mp * mp);
        }

        /// <summary>VMP message to <c>productExp</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>productExp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>productExp</c> as the random arguments are varied. The formula is <c>proj[sum_(b) p(b) factor(productExp,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductExpAverageLogarithm(double A, [SkipIfUniform] Gaussian B)
        {
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            if (Double.IsPositiveInfinity(vb))
                return Gaussian.Uniform();
            double mp = A * Math.Exp(mb + vb / 2.0);
            return Gaussian.FromMeanAndVariance(mp, A * A * Math.Exp(2.0 * (mb + vb)) - mp * mp);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="ProductExp">Incoming message from <c>productExp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>productExp</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_productExp p(productExp) factor(productExp,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="ProductExp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian ProductExp, [Proper] Gaussian B)
        {
            if (B.IsPointMass)
                return GaussianProductVmpOp.AAverageLogarithm(ProductExp, B.Point);
            if (ProductExp.IsPointMass)
                return AAverageLogarithm(ProductExp.Point, B);
            if (!B.IsProper())
                throw new ImproperMessageException(B);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            // catch uniform case to avoid 0*Inf
            if (ProductExp.IsUniform())
                return ProductExp;
            Gaussian result = new Gaussian();
            result.Precision = ProductExp.Precision * Math.Exp(2.0 * mb + 2.0 * vb);
            result.MeanTimesPrecision = ProductExp.MeanTimesPrecision * Math.Exp(mb + vb / 2.0);
            return result;
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="ProductExp">Constant value for <c>productExp</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(productExp,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(double ProductExp, [Proper] Gaussian B)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="ProductExp">Incoming message from <c>productExp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>productExp</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_productExp p(productExp) factor(productExp,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="ProductExp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static NonconjugateGaussian BAverageLogarithm(
            [SkipIfUniform] Gaussian ProductExp, [Proper, SkipIfUniform] Gaussian A, [Proper, SkipIfUniform] Gaussian B, NonconjugateGaussian to_B, NonconjugateGaussian result)
        {
            if (B.IsPointMass)
                return NonconjugateGaussian.Uniform();
            if (ProductExp.IsPointMass)
                return BAverageLogarithm(ProductExp.Point, A);
            if (!B.IsProper())
                throw new ImproperMessageException(B);
            // catch uniform case to avoid 0*Inf
            if (ProductExp.IsUniform())
                return NonconjugateGaussian.Uniform();
            double mx, vx, m, v, mz, vz;
            ProductExp.GetMeanAndVariance(out mz, out vz);
            A.GetMeanAndVariance(out mx, out vx);
            B.GetMeanAndVariance(out m, out v);

            //if (mx * mz < 0)
            //{
            //    Console.WriteLine("Warning: mx*mz < 0, setting to uniform");
            //    result.SetToUniform();
            //    return result;
            //}

            double Ex2 = mx * mx + vx;
            double grad2_S_m2 = -2 * Ex2 * Math.Exp(2 * m + 2 * v) / vz + mx * mz * Math.Exp(m + .5 * v) / vz;
            double grad_m = -Ex2 * Math.Exp(2 * m + 2 * v) / vz + mx * mz * Math.Exp(m + .5 * v) / vz;
            double threshold = 10;
            double mf, vf, afm1, bf;
            if (grad2_S_m2 >= -threshold && mx * mz > 0)
            {
                mf = Math.Log(mx * mz / Ex2) - 1.5 * v;
                vf = (mf - m) / grad_m;
            }
            else
            {
                vf = -1 / grad2_S_m2;
                mf = m - grad_m / grad2_S_m2;
            }

            Gaussian priorG;
            if (result.IsUniform())
                priorG = B;
            else
            {
                var prior = new NonconjugateGaussian();
                prior.SetToRatio((new NonconjugateGaussian(B)), to_B);
                priorG = prior.GetGaussian();
            }

            result.MeanTimesPrecision = mf / vf;
            result.Precision = 1 / vf;

            var updatedM = new Gaussian(mf, vf) * priorG;
            m = updatedM.GetMean();

            double grad_S2_v2 = -2 * Ex2 * Math.Exp(2 * m + 2 * v) / vz + .25 * mx * mz * Math.Exp(m + .5 * v) / vz;
            double grad_S_v = -Ex2 * Math.Exp(2 * m + 2 * v) / vz + .5 * mx * mz * Math.Exp(m + .5 * v) / vz;

            afm1 = -1;
            bf = -1;
            if (grad2_S_m2 >= -threshold)
            {
                afm1 = -v * v * grad_S2_v2;
                bf = -grad_S_v + afm1 / v;
            }

            if ((afm1 < 0 || bf < 0) && mx * mz > 0)
            {
                double v_opt = 2 / 3 * (Math.Log(mx * mz / Ex2 / 2) - m);
                if (v_opt != v)
                {
                    bf = v * grad_S_v / (v_opt - v);
                    afm1 = v_opt * bf;
                }
            }

            if (afm1 < 0 || bf < 0)
            {
                afm1 = -v * v * grad_S2_v2;
                bf = -grad_S_v + afm1 / v;
            }

            if (afm1 < 0 || bf < 0)
            {
                result.Shape = 1;
                result.Rate = 0;
            }
            else
            {
                result.Shape = afm1 + 1;
                result.Rate = bf;
            }

            if (!result.IsProper())
                throw new ApplicationException("improper");
            return result;
            // REMEMBER TO ADD ENTROPY TERM IN REPLICATE OP
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="ProductExp">Incoming message from <c>productExp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>productExp</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_productExp p(productExp) factor(productExp,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="ProductExp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian BAverageLogarithm( // dummy to placate the compiler
            [SkipIfUniform] Gaussian ProductExp, [Proper, SkipIfUniform] Gaussian A, [Proper, SkipIfUniform] Gaussian B, Gaussian result)
        {
            throw new NotImplementedException();
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="ProductExp">Constant value for <c>productExp</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(productExp,a,b)))</c>.</para>
        /// </remarks>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static NonconjugateGaussian BAverageLogarithm(double ProductExp, Gaussian A)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="ProductExp">Incoming message from <c>productExp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>productExp</c> integrated out. The formula is <c>sum_productExp p(productExp) factor(productExp,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="ProductExp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static NonconjugateGaussian BAverageLogarithm(
            [SkipIfUniform] Gaussian ProductExp, double A, [Proper] Gaussian B, NonconjugateGaussian to_B, NonconjugateGaussian result)
        {
            return BAverageLogarithm(ProductExp, Gaussian.PointMass(A), B, to_B, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="ProductExp">Constant value for <c>productExp</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma BAverageLogarithm(double ProductExp, double A)
        {
            return GammaProductOp.BAverageConditional(ProductExp, A);
        }
    }
}
