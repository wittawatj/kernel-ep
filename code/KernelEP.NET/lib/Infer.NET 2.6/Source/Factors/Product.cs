// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Ratio(double, double)" /></description></item><item><description><see cref="Factor.Product(double, double)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double), Default=true)]
    [Quality(QualityBand.Mature)]
    public static class GaussianProductOp
    {
        /// <summary>
        /// The number of quadrature nodes used to compute the messages.
        /// Reduce this number to save time in exchange for less accuracy.
        /// Must be an odd number.
        /// </summary>
        public static int QuadratureNodeCount = 1001; // must be odd to avoid A=0

        /// <summary>
        /// Force proper messages
        /// </summary>
        public static bool ForceProper = true;

        /// <summary />
        /// <param name="A">Incoming message from <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gaussian ProductAverageConditionalInit(Gaussian A, Gaussian B)
        {
            return Gaussian.Uniform();
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>.</param>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(ratio,b) p(ratio,b) factor(ratio,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian ProductAverageConditional(Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return ProductAverageConditional(A.Point, B);
            if (B.IsPointMass)
                return ProductAverageConditional(A, B.Point);
            if (Product.IsPointMass)
                throw new NotImplementedException();
            if (Product.Precision < 1e-100)
                return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
            double z = 0, sumX = 0, sumX2 = 0;
            for (int i = 0; i <= QuadratureNodeCount; i++)
            {
                double a = (2.0 * i) / QuadratureNodeCount - 1;
                double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                double fA = Math.Exp(logfA);

                z += fA;
                double b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                double b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                double x = a * b;
                double x2 = a * a * b2;
                sumX += x * fA;
                sumX2 += x2 * fA;

                double invA = a;
                a = 1.0 / invA;
                double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                double fInvA = Math.Exp(logfInvA);
                z += fInvA;
                b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                x = a * b;
                x2 = a * a * b2;
                sumX += x * fInvA;
                sumX2 += x2 * fInvA;
            }
            double mean = sumX / z;
            double var = sumX2 / z - mean * mean;
            Gaussian result = Gaussian.FromMeanAndVariance(mean, var);
            result.SetToRatio(result, Product, ForceProper);
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[p(ratio) sum_(a,b) p(a,b) factor(ratio,a,b)]/p(ratio)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return AAverageConditional(Product, B.Point);
            if (Product.IsUniform())
                return Gaussian.Uniform();
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            if (A.IsPointMass)
            {
                // f(a) = int_b N(mp; ab, vp) p(b) db
                //      = N(mp; a*mb, vp + a^2*vb)
                // log f(a) = -0.5*log(vp + a^2*vb) -0.5*(mp - a*mb)^2/(vp + a^2*vb)
                // dlogf = -a/(vp + a^2*vb) + mb*(mp - a*mb)/(vp + a^2*vb) + a*vb*(mp - a*mb)^2/(vp + a^2*vb)^2
                // ddlogf = -1/(vp + a^2*vb) + 2*a^2*vb/(vp + a^2*vb)^2 + mb*(mp - mb)/(vp + a^2*vb) + 2*a*vb*mb*(mp - a*mb)/(vp + a^2*vb)^2 + 
                //          vb*(mp - a*mb)^2/(vp + a^2*vb)^2 - 2*a*vb*mb*(mp - a*mb)/(vp + a^2*vb)^2 - (2*a*vb)^2*(mp - a*mb)^2/(vp + a^2*vb)^3
                double denom = vProduct + mA * mA * vB;
                double diff = mProduct - mA * mB;
                double diff2 = diff * diff;
                double denom2 = denom * denom;
                double dlogf = -mA / denom + mB * diff / denom + mA * vB * diff2 / denom2;
                double avb = mA * vB;
                double ddlogf = (mB * (mProduct - mB) - 1) / denom + (2 * mA * avb + vB * diff2) / denom2 - (4 * avb * avb * diff2) / (denom2 * denom);
                return Gaussian.FromDerivatives(mA, dlogf, ddlogf, ForceProper);
            }
            else
            {
                // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
                double z = 0, sumA = 0, sumA2 = 0;
                for (int i = 0; i <= QuadratureNodeCount; i++)
                {
                    double a = (2.0 * i) / QuadratureNodeCount - 1;
                    double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                    double fA = Math.Exp(logfA);
                    z += fA;
                    sumA += a * fA;
                    sumA2 += a * a * fA;

                    double invA = a;
                    a = 1.0 / invA;
                    double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) -
                                      Math.Log(Math.Abs(invA + Double.Epsilon));
                    double fInvA = Math.Exp(logfInvA);
                    z += fInvA;
                    sumA += a * fInvA;
                    sumA2 += a * a * fInvA;
                }
                double mean = sumA / z;
                double var = sumA2 / z - mean * mean;
                if (var <= 0)
                    throw new Exception("quadrature failed");
                Gaussian result = new Gaussian();
                result.SetMeanAndVariance(mean, var);
                result.SetToRatio(result, A, ForceProper);
                return result;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="A">Incoming message from <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[p(ratio) sum_(b) p(b) factor(ratio,a,b)]/p(ratio)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional(double Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return AAverageConditional(Gaussian.PointMass(Product), A, B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a,ratio) p(a,ratio) factor(ratio,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(ratio) p(ratio) factor(ratio,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional(double Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="A">Constant value for <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(b) p(b) factor(ratio,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(ratio) p(ratio) factor(ratio,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageConditional(B, A);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="a">Constant value for <c>ratio</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian ProductAverageConditional(double a, double b)
        {
            return Gaussian.PointMass(a * b);
        }

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[p(ratio) sum_(a) p(a) factor(ratio,a,b)]/p(ratio)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B)
        {
            Gaussian result = new Gaussian();
            if (Product.IsPointMass)
                return AAverageConditional(Product.Point, B);
            // (m - ab)^2/v = (a^2 b^2 - 2abm + m^2)/v
            // This code works correctly even if B=0 or Product is uniform. 
            result.Precision = B * B * Product.Precision;
            result.MeanTimesPrecision = B * Product.MeanTimesPrecision;
            return result;
        }

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>ratio</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian AAverageConditional(double Product, double B)
        {
            if (B == 0)
            {
                if (Product != 0)
                    throw new AllZeroException();
                return Gaussian.Uniform();
            }
            else
                return Gaussian.PointMass(Product / B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>ratio</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a) p(a) factor(ratio,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A)
        {
            return AAverageConditional(Product, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="A">Constant value for <c>ratio</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian BAverageConditional(double Product, double A)
        {
            return AAverageConditional(Product, A);
        }

#if false
		public static double AverageValueLn([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
		{
			double pm, pv;
			Product.GetMeanAndVariance(out pm, out pv);
			// find the variable with most precision.
			Gaussian Wide, Thin;
			if (A.Precision > B.Precision) {
				Wide = B;
				Thin = A;
			} else {
				Wide = A;
				Thin = B;
			}
			if (Thin.IsPointMass) {
				double am, av;
				Wide.GetMeanAndVariance(out am, out av);
				double b = Thin.Point;
				return Gaussian.EvaluateLn(pm, b * am, pv + b * b * av);
			} else {
				// use quadrature to integrate over B
				throw new NotImplementedException();
			}
		}
#endif
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Ratio(double, double)" /></description></item><item><description><see cref="Factor.Product(double, double)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double), Default = true)]
    [FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public static class GaussianProductOp_Slow
    {
        public static int QuadratureNodeCount = 20000;

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>.</param>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(ratio,b) p(ratio,b) factor(ratio,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian ProductAverageConditional(Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A.Point, B);
            if (B.IsPointMass)
                return GaussianProductOp.ProductAverageConditional(A, B.Point);
            if (Product.IsUniform() || Product.Precision < 1e-100)
                return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            bool oldMethod = false;
            if (oldMethod)
            {
                // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
                double z = 0, sumX = 0, sumX2 = 0;
                for (int i = 0; i <= QuadratureNodeCount; i++)
                {
                    double a = (2.0 * i) / QuadratureNodeCount - 1;
                    double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                    double fA = Math.Exp(logfA);

                    z += fA;
                    double b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                    double b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                    double x = a * b;
                    double x2 = a * a * b2;
                    sumX += x * fA;
                    sumX2 += x2 * fA;

                    double invA = a;
                    a = 1.0 / invA;
                    double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                    double fInvA = Math.Exp(logfInvA);
                    z += fInvA;
                    b = (mB * vProduct + a * mProduct * vB) / (vProduct + a * a * vB);
                    b2 = b * b + (vProduct * vB) / (vProduct + a * a * vB);
                    x = a * b;
                    x2 = a * a * b2;
                    sumX += x * fInvA;
                    sumX2 += x2 * fInvA;
                }
                double mean = sumX / z;
                double var = sumX2 / z - mean * mean;
                Gaussian result = Gaussian.FromMeanAndVariance(mean, var);
                result.SetToRatio(result, Product, GaussianProductOp.ForceProper);
                return result;
            }
            else
            {
                double pA = A.Precision;
                double a0, amin, amax;
                GetIntegrationBoundsForA(mProduct, vProduct, mA, pA, mB, vB, out a0, out amin, out amax);
                if (amin == a0 || amax == a0)
                    return AAverageConditional(Product, Gaussian.PointMass(a0), B);
                int n = QuadratureNodeCount;
                double inc = (amax - amin) / (n - 1);
                if (vProduct < 1)
                {
                    // Compute the message directly
                    double Z = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logf = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double v = vProduct + a * a * vB;
                        double diff = mProduct - a * mB;
                        double diffv = diff / v;
                        double diffv2 = diffv * diffv;
                        double v2 = v * v;
                        double dlogf = -diffv;
                        double avb = a * vB;
                        double ddlogf = diffv2 -1/v;
                        if ((i == 0 || i == n - 1) && (logf > -49))
                            throw new Exception("invalid integration bounds");
                        double f = Math.Exp(logf);
                        if (double.IsPositiveInfinity(f))
                        {
                            // this can happen if the likelihood is extremely sharp
                            //throw new Exception("overflow");
                            return ProductAverageConditional(Product, Gaussian.PointMass(a0), B);
                        }
                        Z += f;
                        sum1 += dlogf * f;
                        sum2 += ddlogf * f;
                    }
                    double alpha = sum1 / Z;
                    double beta = alpha*alpha - sum2 / Z;
                    return GaussianProductOp_Laplace.GaussianFromAlphaBeta(Product, alpha, beta, GaussianOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and then divide
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logfA = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double fA = Math.Exp(logfA);
                        double v = vProduct + a * a * vB;
                        double mX = a * (mProduct * a * vB + vProduct * mB)/v;
                        double vX = a*a*vB*vProduct / v;
                        mva.Add(mX, vX, fA);
                    }
                    double mean = mva.Mean;
                    double var = mva.Variance;
                    if (var <= 0)
                        throw new Exception("quadrature failed");
                    Gaussian result = new Gaussian();
                    result.SetMeanAndVariance(mean, var);
                    result.SetToRatio(result, Product, GaussianProductOp.ForceProper);
                    return result;
                }
            }
        }

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[p(ratio) sum_(a,b) p(a,b) factor(ratio,a,b)]/p(ratio)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return GaussianProductOp.AAverageConditional(Product, B.Point);
            if (Product.IsUniform() || B.IsUniform() || Product.Precision < 1e-100)
                return Gaussian.Uniform();
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            if (A.IsPointMass)
            {
                // f(a) = int_b N(mp; ab, vp) p(b) db
                //      = N(mp; a*mb, vp + a^2*vb)
                // log f(a) = -0.5*log(vp + a^2*vb) -0.5*(mp - a*mb)^2/(vp + a^2*vb)
                // dlogf = -a/(vp + a^2*vb) + mb*(mp - a*mb)/(vp + a^2*vb) + a*vb*(mp - a*mb)^2/(vp + a^2*vb)^2
                // ddlogf = -1/(vp + a^2*vb) + 2*a^2*vb/(vp + a^2*vb)^2 + mb*(mp - mb)/(vp + a^2*vb) + 2*a*vb*mb*(mp - a*mb)/(vp + a^2*vb)^2 + 
                //          vb*(mp - a*mb)^2/(vp + a^2*vb)^2 - 2*a*vb*mb*(mp - a*mb)/(vp + a^2*vb)^2 - (2*a*vb)^2*(mp - a*mb)^2/(vp + a^2*vb)^3
                double v = vProduct + mA * mA * vB;
                double diff = mProduct - mA * mB;
                double diffv = diff / v;
                double diffv2 = diffv * diffv;
                double v2 = v * v;
                double dlogf = -mA / v + mB * diffv + mA * vB * diffv2;
                double avb = mA * vB;
                double ddlogf = (mB * (mProduct - mB) - 1) / v + (2 * mA * avb / v2 + vB * diffv2) - (4 * avb * avb * diffv2) / v;
                return Gaussian.FromDerivatives(mA, dlogf, ddlogf, GaussianProductOp.ForceProper);
            }
            else
            {
                double pA = A.Precision;
                double a0, amin, amax;
                GetIntegrationBoundsForA(mProduct, vProduct, mA, pA, mB, vB, out a0, out amin, out amax);
                if (amin == a0 || amax == a0)
                    return AAverageConditional(Product, Gaussian.PointMass(a0), B);
                int n = QuadratureNodeCount;
                double inc = (amax - amin) / (n - 1);
                if (vA < 1)
                {
                    // Compute the message directly
                    double Z = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logf = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double v = vProduct + a * a * vB;
                        double diff = mProduct - a * mB;
                        double diffv = diff / v;
                        double diffv2 = diffv * diffv;
                        double v2 = v * v;
                        double dlogf = -a / v + mB * diffv + a * vB * diffv2;
                        double avb = a * vB;
                        double ddlogf = (mB * (mProduct - mB) - 1) / v + (2 * a * avb / v2 + vB * diffv2) - (4 * avb * avb * diffv2) / v;
                        if ((i == 0 || i == n - 1) && (logf > -49))
                            throw new Exception("invalid integration bounds");
                        double f = Math.Exp(logf);
                        if (double.IsPositiveInfinity(f))
                        {
                            // this can happen if the likelihood is extremely sharp
                            //throw new Exception("overflow");
                            return AAverageConditional(Product, Gaussian.PointMass(a0), B);
                        }
                        Z += f;
                        sum1 += dlogf * f;
                        sum2 += ddlogf * f;
                    }
                    double alpha = sum1 / Z;
                    double beta = alpha * alpha - sum2 / Z;
                    return GaussianProductOp_Laplace.GaussianFromAlphaBeta(A, alpha, beta, GaussianOp.ForceProper);
                }
                else
                {
                    // Compute the marginal and then divide
                    MeanVarianceAccumulator mva = new MeanVarianceAccumulator();
                    for (int i = 0; i < n; i++)
                    {
                        double a = amin + i * inc;
                        double logfA = LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                        double fA = Math.Exp(logfA);
                        mva.Add(a, fA);
                    }
                    double mean = mva.Mean;
                    double var = mva.Variance;
                    if (var <= 0)
                        throw new Exception("quadrature failed");
                    Gaussian result = new Gaussian();
                    result.SetMeanAndVariance(mean, var);
                    result.SetToRatio(result, A, GaussianProductOp.ForceProper);
                    return result;
                }
            }
        }

        public static void GetIntegrationBoundsForA(double mProduct, double vProduct, double mA, double pA, 
            double mB, double vB, out double amode, out double amin, out double amax)
        {
            // this code works even if vA = infinity
            double mpA = mA*pA;
            double vB2 = vB*vB;
            double vProduct2 = vProduct*vProduct;
            // coefficients of polynomial for derivative
            double[] coeffs = { -pA*vB2, mpA*vB2, -pA*2*vProduct*vB-vB2, mpA*2*vProduct*vB -vB*mB*mProduct, 
                                 -pA*vProduct2+vB*mProduct*mProduct-vB*vProduct-vProduct*mB*mB, mpA*vProduct2+vProduct*mB*mProduct };
            //double[] coeffs = { -pA*vB2, mpA*vB2, -pA*2*vProduct*vB, mpA*2*vProduct*vB, 
            //                     -pA*vProduct2, mpA*vProduct2 };
            //Console.WriteLine(StringUtil.CollectionToString(coeffs, " "));
            List<double> stationaryPoints;
            GaussianOp_Slow.GetRealRoots(coeffs, out stationaryPoints);
            // coefficients of polynomial for 2nd derivative
            double[] coeffs2 = new double[7];
            for (int i = 0; i < coeffs2.Length; i++)
            {
                double c = 0;
                if (i >= 2)
                {
                    c += vProduct * coeffs[i - 2] * (5 - (i - 2));
                }
                if (i <= 5)
                {
                    c += vB * coeffs[i] * (5 - i - 4);
                }
                coeffs2[i] = c;
            }
            //Console.WriteLine(StringUtil.CollectionToString(coeffs2, " "));
            List<double> inflectionPoints;
            GaussianOp_Slow.GetRealRoots(coeffs2, out inflectionPoints);
            Func<double, double> like = a => LogLikelihood(a, mProduct, vProduct, mA, pA, mB, vB);
            var stationaryValues = stationaryPoints.ConvertAll(a => like(a));
            double max = MMath.Max(stationaryValues);
            double a0 = stationaryPoints[stationaryValues.IndexOf(max)];
            amode = a0;
            Func<double, double> func = a =>
            {
                return LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB) + 50;
            };
            Func<double, double> deriv = a =>
            {
                if (double.IsInfinity(a))
                    return -a;
                double v = vProduct + vB * a * a;
                double diffv = (mProduct - a * mB)/v;
                double diffa = a - mA;
	            return a*vB*(diffv*diffv -1/v) + mB*diffv - diffa * pA;
            };
            // find where the likelihood matches the bound value
            List<double> zeroes = GaussianOp_Slow.FindZeroes(func, deriv, stationaryPoints, inflectionPoints);
            amin = MMath.Min(zeroes);
            amax = MMath.Max(zeroes);
            Assert.IsTrue(amin < amax);
            //Console.WriteLine("amin = {0} amode = {1} amax = {2}", amin, amode, amax);
        }

        internal static double LogLikelihood(double a, double mProduct, double vProduct, double mA, double pA, double mB, double vB)
        {
            if (double.IsInfinity(a))
                return double.NegativeInfinity;
            double v = vProduct + vB * a * a;
            double diff = mProduct - a * mB;
            double diffa = a - mA;
            return -0.5 * (Math.Log(v) + diff * diff / v + diffa * diffa * pA);
        }

        internal static double LogLikelihoodRatio(double a, double a0, double mProduct, double vProduct, double mA, double pA, double mB, double vB)
        {
            if (double.IsInfinity(a))
                return double.NegativeInfinity;
            double v = vProduct + vB * a * a;
            double diff = mProduct - a * mB;
            double diffa = a - mA;
            double v0 = vProduct + vB * a0 * a0;
            double diff0 = mProduct - a0 * mB;
            double diffa0 = a0 - mA;
            return -0.5 * (Math.Log(v / v0) + diff * diff / v + diffa * diffa * pA) + 0.5 * (diff0 * diff0 / v0 + diffa0 * diffa0 * pA);
        }

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="A">Incoming message from <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[p(ratio) sum_(b) p(b) factor(ratio,a,b)]/p(ratio)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian AAverageConditional(double Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return AAverageConditional(Gaussian.PointMass(Product), A, B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a,ratio) p(a,ratio) factor(ratio,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(ratio) p(ratio) factor(ratio,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gaussian BAverageConditional(double Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Product(double, double)" /></description></item><item><description><see cref="Factor.Product_SHG09(double, double)" /></description></item></list>, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor as if running VMP, as required by Stern's algorithm.
    /// The algorithm comes from "Matchbox: Large Scale Online Bayesian Recommendations" by David Stern, Ralf Herbrich, and Thore Graepel, WWW 2009.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product_SHG09", typeof(double), typeof(double))]
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Preview)]
    public static class GaussianProductOp_SHG09
    {
        public static Gaussian ProductAverageConditional2(
            Gaussian product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, [NoInit] Gaussian to_A, [NoInit] Gaussian to_B)
        {
            // this version divides out the message from product, so the marginal for Product is correct and the factor is composable.
            return GaussianProductVmpOp.ProductAverageLogarithm(A * to_A, B * to_B) / product;
        }

        // FIXME: the Cancels attributes here are not correct but are a temporary hack to get the Recommender Learner to have a correct schedule
        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, [NoInit, Cancels] Gaussian to_A, [NoInit, Cancels] Gaussian to_B)
        {
            // note we are not dividing out the message from Product.
            // this means that the marginal for Product will not be correct, and the factor is not composable.
            return GaussianProductVmpOp.ProductAverageLogarithm(A * to_A, B * to_B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [Proper /*, Fresh*/] Gaussian B, [NoInit] Gaussian to_B)
        {
            return GaussianProductVmpOp.AAverageLogarithm(Product, B * to_B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [Proper /*, Fresh*/] Gaussian A, [NoInit] Gaussian to_A)
        {
            //return BAverageConditional(Product, A.GetMean());
            return AAverageConditional(Product, A, to_A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <param name="to_product">Previous outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio(
            [SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A, Gaussian to_B, Gaussian to_product)
        {
            // The SHG paper did not define how to compute evidence.
            // The formula below comes from matching the VMP evidence for a model with a single product factor.
            Gaussian qA = A * to_A;
            Gaussian qB = B * to_B;
            Gaussian qProduct = to_product;
            double aRatio = A.GetLogAverageOf(to_A) - qA.GetAverageLog(to_A);
            double bRatio = B.GetLogAverageOf(to_B) - qB.GetAverageLog(to_B);
            double productRatio = qProduct.GetAverageLog(Product) - to_product.GetLogAverageOf(Product);
            return aRatio + bRatio + productRatio;
        }

#if true
        // FIXME: the Cancels attributes here are not correct but are a temporary hack to get the Recommender Learner to have a correct schedule
        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(b) p(b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B, [NoInit, Cancels] Gaussian to_B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B * to_B);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a) p(a) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B, [NoInit, Cancels] Gaussian to_A)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A * to_A, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product) p(product) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B)
        {
            return GaussianProductVmpOp.AAverageLogarithm(Product, B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A)
        {
            return AAverageConditional(Product, A);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <param name="to_product">Previous outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a) p(product,a) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, double B, Gaussian to_A, Gaussian to_product)
        {
            // The SHG paper did not define how to compute evidence.
            // The formula below comes from matching the VMP evidence for a model with a single product factor.
            Gaussian qA = A * to_A;
            Gaussian qProduct = to_product;
            double aRatio = A.GetLogAverageOf(to_A) - qA.GetAverageLog(to_A);
            double productRatio = qProduct.GetAverageLog(Product) - to_product.GetLogAverageOf(Product);
            return aRatio + productRatio;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <param name="to_product">Previous outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,b) p(product,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, double A, [SkipIfUniform] Gaussian B, Gaussian to_B, Gaussian to_product)
        {
            // The SHG paper did not define how to compute evidence.
            // The formula below comes from matching the VMP evidence for a model with a single product factor.
            Gaussian qB = B * to_B;
            Gaussian qProduct = to_product;
            double bRatio = B.GetLogAverageOf(to_B) - qB.GetAverageLog(to_B);
            double productRatio = qProduct.GetAverageLog(Product) - to_product.GetLogAverageOf(Product);
            return bRatio + productRatio;
        }
#else
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductOp.ProductAverageConditional(A, B);
        }

        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B)
        {
            return GaussianProductOp.ProductAverageConditional(A, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product) p(product) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B)
        {
            return GaussianProductOp.AAverageConditional(Product, B);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A)
        {
            return GaussianProductOp.BAverageConditional(Product, A);
        }

        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, double B)
        {
            return GaussianProductEvidenceOp.LogEvidenceRatio(Product, A, B);
        }

        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, double A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductEvidenceOp.LogEvidenceRatio(Product, A, B);
        }
#endif
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor using Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class GaussianProductOp_Laplace2
    {
        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        public static Gaussian ProductAverageConditional(Gaussian A, Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        public static Gaussian ProductAverageConditional2(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            if (Product.IsUniform())
                return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double denom = 1 / (vx + ahat * ahat * vb);
            double diff = mx - ahat * mb;
            double dlogz = -diff * denom;
            double dd = ahat * vb;
            double q = denom * (mb + 2 * dd * diff * denom);
            double ddd = vb;
            double n = diff * diff;
            double dn = -diff * mb;
            double ddn = mb * mb;
            double dda = denom * (-(ddd + ddn) + denom * (2 * dd * (dd + 2 * dn) + n * ddd - denom * 4 * n * dd * dd));
            double da = va * q / (1 - va * dda);
            double ddlogz = -denom + q * da;
            return GaussianProductOp_Laplace.GaussianFromAlphaBeta(Product, dlogz, -ddlogz, true);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        public static Gaussian AAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            Gaussian Apost = A * to_A;
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ahat = Apost.GetMean();
            //ahat = A.GetMean();
            double denom = 1 / (vx + ahat * ahat * vb);
            double diff = mx - ahat * mb;
            double dd = ahat * vb;
            double ddd = vb;
            double n = diff * diff;
            double dn = -diff * mb;
            double ddn = mb * mb;
            double dlogf = denom * (-(dd + dn) + denom * n * dd);
            double ddlogf = denom * (-(ddd + ddn) + denom * (2 * dd * (dd + 2 * dn) + n * ddd - denom * 4 * n * dd * dd));
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * ahat + dlogf, r);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static Gaussian BAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_B)
        {
            return AAverageConditional(Product, B, A, to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A)
        {
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            return Gaussian.GetLogProb(mx, ahat * mb, vx + ahat * ahat * vb) + A.GetLogProb(ahat) - Apost.GetLogProb(ahat);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <param name="to_product">Previous outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A, Gaussian to_product)
        {
            return LogAverageFactor(Product, A, B, to_A) - to_product.GetLogAverageOf(Product);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor using Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class GaussianProductOp_Laplace
    {
        public static bool modified = true;
        public static bool offDiagonal = false;

        public static Gaussian GaussianFromAlphaBeta(Gaussian prior, double alpha, double beta, bool forceProper)
        {
            if (prior.IsPointMass)
                return Gaussian.FromDerivatives(prior.Point, alpha, -beta, forceProper);
            double prec = prior.Precision;
            double tau = prior.MeanTimesPrecision;
            double weight = beta / (prec - beta);
            if (forceProper && (prec == beta || weight < 0))
                weight = 0;
            // eq (31) in EP quickref; same as inv(inv(beta)-inv(prec))
            double resultPrecision = prec * weight;
            // eq (30) in EP quickref times above and simplified
            double resultMeanTimesPrecision = weight * (tau + alpha) + alpha;
            if (double.IsNaN(resultPrecision) || double.IsNaN(resultMeanTimesPrecision))
                throw new ApplicationException("result is nan");
            return Gaussian.FromNatural(resultMeanTimesPrecision, resultPrecision);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        public static Gaussian ProductAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A, Gaussian to_B)
        {
            if (Product.IsUniform())
            {
                double ma, va;
                A.GetMeanAndVariance(out ma, out va);
                double mb, vb;
                B.GetMeanAndVariance(out mb, out vb);
                // this is the limiting value of below - it excludes va*vb
                return Gaussian.FromMeanAndVariance(ma * mb, ma * ma * vb + mb * mb * va);
            }
            Gaussian Apost = A * to_A;
            Gaussian Bpost = B * to_B;
            double ahat = Apost.GetMean();
            double bhat = Bpost.GetMean();
            double gx = -(Product.MeanTimesPrecision - ahat * bhat * Product.Precision);
            double gxx = -Product.Precision;
            double gax = bhat * Product.Precision;
            double gbx = ahat * Product.Precision;
            double gaa = -bhat * bhat * Product.Precision;
            double gbb = -ahat * ahat * Product.Precision;
            double gab = Product.MeanTimesPrecision - 2 * ahat * bhat * Product.Precision;
            if (modified)
            {
                double gaaa = 0;
                double gaaab = 0;
                double gaaaa = 0;
                double gaab = -2 * bhat * Product.Precision;
                double gaabb = -2 * Product.Precision;
                double gabb = -2 * ahat * Product.Precision;
                double gabbb = 0;
                double gbbb = 0;
                double gbbbb = 0;
                double adiff = A.Precision - gaa;
                double bdiff = B.Precision - gbb;
                double h = adiff * bdiff;
                double ha = -gaaa * bdiff - gabb * adiff;
                double hb = -gaab * bdiff - gbbb * adiff;
                double haa = -gaaaa * bdiff - gaabb * adiff + 2 * gaaa * gabb;
                double hab = -gaaab * bdiff - gabbb * adiff + gaaa * gbbb + gabb * gaab;
                double hbb = -gaabb * bdiff - gbbbb * adiff + 2 * gaab * gbbb;
                if (offDiagonal)
                {
                    h += -gab * gab;
                    ha += -2 * gab * gaab;
                    hb += -2 * gab * gabb;
                    haa += -2 * gaab * gaab - 2 * gab * gaaab;
                    hab += -2 * gabb * gaab - 2 * gab * gaabb;
                    hbb += -2 * gabb * gabb - 2 * gab * gabbb;
                }
                double logha = ha / h;
                double loghb = hb / h;
                double loghaa = haa / h - logha * logha;
                double loghab = hab / h - logha * loghb;
                double loghbb = hbb / h - loghb * loghb;
                gaa -= 0.5 * loghaa;
                gab -= 0.5 * loghab;
                gbb -= 0.5 * loghbb;
            }
            double cb = gab / (B.Precision - gbb);
            double dax = (gax + gbx * cb) / (A.Precision - (gaa + gab * cb));
            double dbx = gbx / (B.Precision - gbb) + cb * dax;
            double dlogz = gx;
            double ddlogz = gxx + gax * dax + gbx * dbx;
            return GaussianProductOp_Laplace.GaussianFromAlphaBeta(Product, dlogz, -ddlogz, true);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        public static Gaussian AAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            double bhat = (ahat * Product.MeanTimesPrecision + B.MeanTimesPrecision) / (ahat * ahat * Product.Precision + B.Precision);
            double ga = (Product.MeanTimesPrecision - ahat * bhat * Product.Precision) * bhat;
            double gaa = -bhat * bhat * Product.Precision;
            double gbb = -ahat * ahat * Product.Precision;
            double gab = Product.MeanTimesPrecision - 2 * ahat * bhat * Product.Precision;
            if (modified)
            {
                double gaaa = 0;
                double gaaab = 0;
                double gaaaa = 0;
                double gaab = -2 * bhat * Product.Precision;
                double gaabb = -2 * Product.Precision;
                double gabb = -2 * ahat * Product.Precision;
                double gabbb = 0;
                double gbbb = 0;
                double gbbbb = 0;
                double adiff = A.Precision - gaa;
                double bdiff = B.Precision - gbb;
                double h = adiff * bdiff;
                double ha = -gaaa * bdiff - gabb * adiff;
                double hb = -gaab * bdiff - gbbb * adiff;
                double haa = -gaaaa * bdiff - gaabb * adiff + 2 * gaaa * gabb;
                double hab = -gaaab * bdiff - gabbb * adiff + gaaa * gbbb + gabb * gaab;
                double hbb = -gaabb * bdiff - gbbbb * adiff + 2 * gaab * gbbb;
                if (offDiagonal)
                {
                    // this can cause h to be negative
                    h += -gab * gab;
                    ha += -2 * gab * gaab;
                    hb += -2 * gab * gabb;
                    haa += -2 * gaab * gaab - 2 * gab * gaaab;
                    hab += -2 * gabb * gaab - 2 * gab * gaabb;
                    hbb += -2 * gabb * gabb - 2 * gab * gabbb;
                }
                double logha = ha / h;
                double loghb = hb / h;
                double loghaa = haa / h - logha * logha;
                double loghab = hab / h - logha * loghb;
                double loghbb = hbb / h - loghb * loghb;
                ga -= 0.5 * logha;
                gaa -= 0.5 * loghaa;
                gab -= 0.5 * loghab;
                gbb -= 0.5 * loghbb;
            }
            double cb = gab / (B.Precision - gbb);
            double ddlogf = gaa + cb * gab;
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * ahat + ga, r);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static Gaussian BAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_B)
        {
            return AAverageConditional(Product, B, A, to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, Gaussian to_A)
        {
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            Gaussian Apost = A * to_A;
            double ahat = Apost.GetMean();
            return Gaussian.GetLogProb(mx, ahat * mb, vx + ahat * ahat * vb) + A.GetLogProb(ahat) - Apost.GetLogProb(ahat);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            // TODO
            return 0.0;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor using modified Laplace's method.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    internal static class GaussianProductOp6
    {
        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        public static Gaussian ProductAverageConditional(Gaussian A, Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        public static Gaussian AAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_A)
        {
            Gaussian Apost = A * to_A;
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double ma, va;
            bool usePost = false;
            if (usePost)
                Apost.GetMeanAndVariance(out ma, out va);
            else
                A.GetMeanAndVariance(out ma, out va);
            double ahat = ma;
            ahat = Apost.GetMean();
            double denom = vx + ahat * ahat * vb;
            double diff = mx - ahat * mb;
            double dd = 2 * ahat * vb;
            double ddd = 2 * vb;
            double n = diff * diff;
            double dn = -2 * diff * mb;
            double ddn = 2 * mb * mb;
            double dlogf = -0.5 * (dd + dn) / denom + 0.5 * n * dd / (denom * denom);
            double ddlogf = -0.5 * (ddd + ddn) / denom + 0.5 * (dd * dd + 2 * dd * dn + n * ddd) / (denom * denom) - n * dd * dd / (denom * denom * denom);
            if (usePost)
            {
                double m0, v0;
                A.GetMeanAndVariance(out m0, out v0);
                dlogf += (m0 - ahat) / v0;
                ddlogf += 1 / va - 1 / v0;
                // at a fixed point, dlogf2 = 0 so (ahat - m0)/v0 = dlogf, ahat = m0 + v0*dlogf
                // this is the same fixed point condition as Laplace
            }
            // Ef'/Ef =approx dlogf*f(ahat)/f(ahat) = dlogf
            double mnew = ma + va * dlogf;
            double vnew = va * (1 + va * (ddlogf - dlogf * dlogf));
            double rnew = Math.Max(1 / vnew, A.Precision);
            return Gaussian.FromNatural(rnew * mnew - A.MeanTimesPrecision, rnew - A.Precision);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static Gaussian BAverageConditional(Gaussian Product, Gaussian A, Gaussian B, Gaussian to_B)
        {
            return AAverageConditional(Product, B, A, to_B);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            // TODO
            return 0.0;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor as a linear factor.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    [Buffers("weights")]
    internal static class GaussianProductOp3
    {
        /// <summary>Update the buffer <c>weights</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>New value of buffer <c>weights</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Vector Weights(double A, Gaussian B)
        {
            Vector weights = Vector.Zero(4);
            weights[1] = A;
            return weights;
        }

        /// <summary>Update the buffer <c>weights</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>New value of buffer <c>weights</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Vector Weights(Gaussian A, double B)
        {
            Vector weights = Vector.Zero(4);
            weights[0] = B;
            return weights;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>Update the buffer <c>weights</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_A">Previous outgoing message to <c>A</c>.</param>
        /// <param name="to_B">Previous outgoing message to <c>B</c>.</param>
        /// <returns>New value of buffer <c>weights</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Vector Weights(Gaussian A, Gaussian B, Gaussian to_A, Gaussian to_B)
        {
            if (A.IsPointMass)
                return Weights(A.Point, B);
            if (B.IsPointMass)
                return Weights(A, B.Point);
            A *= to_A;
            B *= to_B;
            double ma, va, mb, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            double ma2 = va + ma * ma;
            double mb2 = vb + mb * mb;
            Vector w = Vector.Zero(3);
            w[0] = ma2 * mb;
            w[1] = mb2 * ma;
            w[2] = ma * mb;
            PositiveDefiniteMatrix M = new PositiveDefiniteMatrix(3, 3);
            M[0, 0] = ma2;
            M[0, 1] = ma * mb;
            M[0, 2] = ma;
            M[1, 0] = ma * mb;
            M[1, 1] = mb2;
            M[1, 2] = mb;
            M[2, 0] = ma;
            M[2, 1] = mb;
            M[2, 2] = 1;
            w = w.PredivideBy(M);
            Vector weights = Vector.Zero(4);
            weights[0] = w[0];
            weights[1] = w[1];
            weights[2] = w[2];
            weights[3] = ma2 * mb2 - w[0] * ma2 * mb - w[1] * mb2 * ma - w[2] * ma * mb;
            if (weights[3] < 0)
                weights[3] = 0;
            if (false)
            {
                // debugging
                GaussianEstimator est = new GaussianEstimator();
                for (int i = 0; i < 10000; i++)
                {
                    double sa = A.Sample();
                    double sb = B.Sample();
                    double f = sa * sb;
                    double g = sa * weights[0] + sb * weights[1] + weights[2];
                    est.Add(f - g);
                }
                Console.WriteLine(weights);
                Console.WriteLine(est.GetDistribution(new Gaussian()));
            }
            return weights;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B, [Fresh] Vector weights)
        {
            // factor is product = N(w[0]*a + w[1]*b + w[2], w[3])
            double v = weights[3];
            Gaussian m = DoublePlusOp.SumAverageConditional(GaussianProductOp.ProductAverageConditional(weights[0], A),
                                                            GaussianProductOp.ProductAverageConditional(weights[1], B));
            m = DoublePlusOp.SumAverageConditional(m, weights[2]);
            return GaussianFromMeanAndVarianceOp.SampleAverageConditional(m, v);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(b) p(b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B, [Fresh] Vector weights)
        {
            return ProductAverageConditional(Gaussian.PointMass(A), B, weights);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a) p(a) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B, [Fresh] Vector weights)
        {
            return ProductAverageConditional(A, Gaussian.PointMass(B), weights);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian B, [Fresh] Vector weights)
        {
            // factor is product = N(w[0]*a + w[1]*b + w[2], w[3])
            double v = weights[3];
            Gaussian sum_B = GaussianFromMeanAndVarianceOp.MeanAverageConditional(Product, v);
            sum_B = DoublePlusOp.AAverageConditional(sum_B, weights[2]);
            Gaussian scale_B = DoublePlusOp.AAverageConditional(sum_B, GaussianProductOp.ProductAverageConditional(weights[1], B));
            return GaussianProductOp.AAverageConditional(scale_B, weights[0]);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product) p(product) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, double B, [Fresh] Vector weights)
        {
            return AAverageConditional(Product, Gaussian.PointMass(B), weights);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [Fresh] Vector weights)
        {
            // factor is product = N(w[0]*a + w[1]*b + w[2], w[3])
            double v = weights[3];
            Gaussian sum_B = GaussianFromMeanAndVarianceOp.MeanAverageConditional(Product, v);
            sum_B = DoublePlusOp.AAverageConditional(sum_B, weights[2]);
            Gaussian scale_B = DoublePlusOp.AAverageConditional(sum_B, GaussianProductOp.ProductAverageConditional(weights[0], A));
            return GaussianProductOp.AAverageConditional(scale_B, weights[1]);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="weights">Buffer <c>weights</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A, [Fresh] Vector weights)
        {
            return BAverageConditional(Product, Gaussian.PointMass(A), weights);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return 0.0;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor using an approximation to the integral Z.
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    internal static class GaussianProductOp4
    {
        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(b) p(b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(double A, [SkipIfUniform] Gaussian B)
        {
            return ProductAverageConditional(Gaussian.PointMass(A), B);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a) p(a) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional([SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageConditional(A, Gaussian.PointMass(B));
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return Gaussian.Uniform();
            double ma, va;
            A.GetMeanAndVariance(out ma, out va);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            double mx, vx;
            Product.GetMeanAndVariance(out mx, out vx);
            double diff = mx - ma * mb;
            double prec = 1 / (vx + va * vb + va * mb * mb + vb * ma * ma);
            //if (prec < 1e-14) return Gaussian.Uniform();
            double alpha = prec * (-vb * ma + prec * diff * diff * ma * vb + diff * mb);
            double beta = alpha * alpha + prec * (1 - diff * diff * prec) * (vb + mb * mb);
            //if (beta == 0) return Gaussian.Uniform();
            if (double.IsNaN(alpha) || double.IsNaN(beta))
                throw new Exception("alpha is nan");
            double r = beta / (A.Precision - beta);
            Gaussian result = new Gaussian();
            result.Precision = r * A.Precision;
            result.MeanTimesPrecision = r * (alpha + A.MeanTimesPrecision) + alpha;
            //Gaussian result = new Gaussian(ma + alpha/beta, 1/beta - va);
            if (double.IsNaN(result.Precision) || double.IsNaN(result.MeanTimesPrecision))
                throw new Exception("result is nan");
            return result;
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product) p(product) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, double B)
        {
            return AAverageConditional(Product, A, Gaussian.PointMass(B));
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A, Gaussian B)
        {
            return BAverageConditional(Product, B, Gaussian.PointMass(A));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return 0.0;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    /// <remarks>
    /// This class allows EP to process the product factor using a log-normal approximation to the input distributions
    /// </remarks>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    internal static class GaussianProductOp5
    {
        public static Gaussian GetExpMoments(Gaussian x)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            return new Gaussian(Math.Exp(m + v / 2), Math.Exp(2 * m + v) * (Math.Exp(v) - 1));
        }

        public static Gaussian GetLogMoments(Gaussian x)
        {
            double m, v;
            x.GetMeanAndVariance(out m, out v);
            double lv = Math.Log(v / (m * m) + 1);
            double lm = Math.Log(Math.Abs(m)) - lv / 2;
            return new Gaussian(lm, lv);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a,b) p(a,b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            Gaussian logA = GetLogMoments(A);
            Gaussian logB = GetLogMoments(B);
            Gaussian logMsg = DoublePlusOp.SumAverageConditional(logA, logB);
            if (false)
            {
                Gaussian logProduct = GetLogMoments(Product);
                Gaussian logPost = logProduct * logMsg;
                return GetExpMoments(logPost) / Product;
            }
            else
            {
                return GetExpMoments(logMsg);
            }
            //return GaussianProductVmpOp.ProductAverageLogarithm(A, B);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(b) p(b) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(Gaussian Product, double A, [SkipIfUniform] Gaussian B)
        {
            return ProductAverageConditional(Product, Gaussian.PointMass(A), B);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a) p(a) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageConditional(Gaussian Product, [SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageConditional(Product, A, Gaussian.PointMass(B));
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product,b) p(product,b) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, [SkipIfUniform] Gaussian B)
        {
            if (A.IsPointMass)
                return Gaussian.Uniform();
            Gaussian logA = GetLogMoments(A);
            Gaussian logB = GetLogMoments(B);
            Gaussian logProduct = GetLogMoments(Product);
            Gaussian logMsg = DoublePlusOp.AAverageConditional(logProduct, logB);
            return GetExpMoments(logMsg);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product) p(product) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageConditional([SkipIfUniform] Gaussian Product, Gaussian A, double B)
        {
            return AAverageConditional(Product, A, Gaussian.PointMass(B));
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product,a) p(product,a) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, Gaussian B)
        {
            return AAverageConditional(Product, B, A);
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageConditional([SkipIfUniform] Gaussian Product, double A, Gaussian B)
        {
            return BAverageConditional(Product, B, Gaussian.PointMass(A));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, [SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            return 0.0;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double), Default = true)]
    [Quality(QualityBand.Mature)]
    public static class GaussianProductEvidenceOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_product">Previous outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] Gaussian Product, Gaussian A, Gaussian B, Gaussian to_product)
        {
            if (A.IsPointMass)
                return GaussianProductEvidenceOp.LogAverageFactor(Product, B, A.Point, to_product);
            if (B.IsPointMass)
                return GaussianProductEvidenceOp.LogAverageFactor(Product, A, B.Point, to_product);
            if (Product.IsUniform())
                return 0.0;
            double mA, vA;
            A.GetMeanAndVariance(out mA, out vA);
            double mB, vB;
            B.GetMeanAndVariance(out mB, out vB);
            double mProduct, vProduct;
            Product.GetMeanAndVariance(out mProduct, out vProduct);
            bool oldMethod = true;
            if (oldMethod)
            {
                // algorithm: quadrature on A from -1 to 1, plus quadrature on 1/A from -1 to 1.
                double z = 0;
                for (int i = 0; i <= GaussianProductOp.QuadratureNodeCount; i++)
                {
                    double a = (2.0 * i) / GaussianProductOp.QuadratureNodeCount - 1;
                    double logfA = Gaussian.GetLogProb(mProduct, a * mB, vProduct + a * a * vB) + Gaussian.GetLogProb(a, mA, vA);
                    double fA = Math.Exp(logfA);
                    z += fA;

                    double invA = a;
                    a = 1.0 / invA;
                    double logfInvA = Gaussian.GetLogProb(mProduct * invA, mB, vProduct * invA * invA + vB) + Gaussian.GetLogProb(a, mA, vA) - Math.Log(Math.Abs(invA + Double.Epsilon));
                    double fInvA = Math.Exp(logfInvA);
                    z += fInvA;
                }
                double inc = 2.0 / GaussianProductOp.QuadratureNodeCount;
                return Math.Log(z * inc);
            }
            else
            {
                double pA = A.Precision;
                double a0, amin, amax;
                GaussianProductOp_Slow.GetIntegrationBoundsForA(mProduct, vProduct, mA, pA, mB, vB, out a0, out amin, out amax);
                if (amin == a0 || amax == a0)
                    return GaussianProductEvidenceOp.LogAverageFactor(Product, B, a0, to_product);
                int n = GaussianProductOp_Slow.QuadratureNodeCount;
                double inc = (amax - amin) / (n - 1);
                double logZ = GaussianProductOp_Slow.LogLikelihood(a0, mProduct, vProduct, mA, pA, mB, vB);
                logZ += 0.5*Math.Log(pA) - 2*MMath.LnSqrt2PI;
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double a = amin + i * inc;
                    double logfA = GaussianProductOp_Slow.LogLikelihoodRatio(a, a0, mProduct, vProduct, mA, pA, mB, vB);
                    double fA = Math.Exp(logfA);
                    sum += fA;
                }
                return logZ + Math.Log(sum * inc);
            }
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="to_product">Previous outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a,b) p(product,a,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gaussian Product, Gaussian A, Gaussian B, Gaussian to_product)
        {
            return LogAverageFactor(Product, A, B, to_product) - to_product.GetLogAverageOf(Product);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <param name="to_product">Outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a) p(product,a) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian product, Gaussian a, double b, [Fresh] Gaussian to_product)
        {
            //Gaussian to_product = GaussianProductOp.ProductAverageConditional(a, b);
            return to_product.GetLogAverageOf(product);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <param name="to_product">Outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,b) p(product,b) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian product, double a, Gaussian b, [Fresh] Gaussian to_product)
        {
            return LogAverageFactor(product, b, a, to_product);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double product, Gaussian a, double b)
        {
            Gaussian to_product = GaussianProductOp.ProductAverageConditional(a, b);
            return to_product.GetLogProb(product);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double product, double a, Gaussian b)
        {
            return LogAverageFactor(product, b, a);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double product, double a, double b)
        {
            return (product == Factor.Product(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double product, double a, double b)
        {
            return LogAverageFactor(product, a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product) p(product) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian product, double a, double b)
        {
            return product.GetLogProb(Factor.Product(a, b));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a) p(product,a) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Gaussian product, Gaussian a, double b)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,b) p(product,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Gaussian product, double a, Gaussian b)
        {
            return LogEvidenceRatio(product, b, a);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double product, Gaussian a, double b)
        {
            return LogAverageFactor(product, a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double product, double a, Gaussian b)
        {
            return LogEvidenceRatio(product, b, a);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Ratio(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Ratio", typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class GaussianRatioEvidenceOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Incoming message from <c>ratio</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <param name="to_ratio">Outgoing message to <c>ratio</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(ratio,a) p(ratio,a) factor(ratio,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian ratio, Gaussian a, double b, [Fresh] Gaussian to_ratio)
        {
            //Gaussian to_ratio = GaussianProductOp.AAverageConditional(a, b);
            return to_ratio.GetLogAverageOf(ratio);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Incoming message from <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <param name="to_ratio">Outgoing message to <c>ratio</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(ratio,b) p(ratio,b) factor(ratio,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian ratio, double a, Gaussian b, [Fresh] Gaussian to_ratio)
        {
            return LogAverageFactor(ratio, b, a, to_ratio);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(ratio,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double ratio, Gaussian a, double b)
        {
            Gaussian to_ratio = GaussianProductOp.AAverageConditional(a, b);
            return to_ratio.GetLogProb(ratio);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(ratio,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double ratio, double a, Gaussian b)
        {
            return LogAverageFactor(ratio, b, a);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(ratio,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double ratio, double a, double b)
        {
            return (ratio == Factor.Ratio(a, b)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(ratio,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double ratio, double a, double b)
        {
            return LogAverageFactor(ratio, a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Incoming message from <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(ratio) p(ratio) factor(ratio,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian ratio, double a, double b)
        {
            return ratio.GetLogProb(Factor.Ratio(a, b));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Incoming message from <c>ratio</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(ratio,a) p(ratio,a) factor(ratio,a,b) / sum_ratio p(ratio) messageTo(ratio))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Gaussian ratio, Gaussian a, double b)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Incoming message from <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(ratio,b) p(ratio,b) factor(ratio,a,b) / sum_ratio p(ratio) messageTo(ratio))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Gaussian ratio, double a, Gaussian b)
        {
            return LogEvidenceRatio(ratio, b, a);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(ratio,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double ratio, Gaussian a, double b)
        {
            return LogAverageFactor(ratio, a, b);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="a">Constant value for <c>a</c>.</param>
        /// <param name="b">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(ratio,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double ratio, double a, Gaussian b)
        {
            return LogEvidenceRatio(ratio, b, a);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class GaussianProductVmpOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(Gaussian product)
        {
            return 0.0;
        }

        internal const string NotSupportedMessage = "Variational Message Passing does not support a Product factor with fixed output and two random inputs.";

        /// <summary>VMP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(product,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageLogarithm([SkipIfUniform] Gaussian A, [SkipIfUniform] Gaussian B)
        {
            Gaussian result = new Gaussian();
            // p(x|a,b) = N(E[a]*E[b], E[b]^2*var(a) + E[a]^2*var(b) + var(a)*var(b))
            double ma, va, mb, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            // Uses John Winn's rule for deterministic factors.
            // Strict variational inference would set the variance to 0.
            result.SetMeanAndVariance(ma * mb, mb * mb * va + ma * ma * vb + va * vb);
            return result;
        }

        public static Gaussian ProductDeriv(Gaussian Product, [SkipIfUniform, Stochastic] Gaussian A, [SkipIfUniform, Stochastic] Gaussian B, Gaussian to_A, Gaussian to_B)
        {
            if (A.IsPointMass)
                return ProductDeriv(Product, A.Point, B, to_B);
            if (B.IsPointMass)
                return ProductDeriv(Product, A, B.Point, to_A);
            double ma, va, mb, vb;
            A.GetMeanAndVariance(out ma, out va);
            B.GetMeanAndVariance(out mb, out vb);
            //Console.WriteLine("ma = {0}, va = {1}, mb = {2}, vb = {3}", ma, va, mb, vb);
            double ma0, va0, mb0, vb0;
            (A / to_A).GetMeanAndVariance(out ma0, out va0);
            (B / to_B).GetMeanAndVariance(out mb0, out vb0);
            Gaussian to_A2 = AAverageLogarithm(Product, B);
            double va2 = 1 / (1 / va0 + to_A2.Precision);
            double ma2 = va2 * (ma0 / va0 + to_A2.MeanTimesPrecision);
            Gaussian to_B2 = BAverageLogarithm(Product, A);
            double vb2 = 1 / (1 / vb0 + to_B2.Precision);
            double mb2 = vb2 * (mb0 / vb0 + to_B2.MeanTimesPrecision);
            double dva2 = 0;
            double dma2 = va2 * mb + dva2 * ma2 / va2;
            double dvb2 = 0;
            // this doesn't seem to help
            //dvb2 = -vb2*vb2*2*ma2*dma2*Product.Precision;
            double dmb2 = vb2 * ma + dvb2 * mb2 / vb2;
            double pPrec2 = 1 / (va2 * vb2 + va2 * mb2 * mb2 + vb2 * ma2 * ma2);
            double dpPrec2 = -(dva2 * vb2 + va2 * dvb2 + dva2 * mb2 * mb2 + va2 * 2 * mb2 * dmb2 + dvb2 * ma2 * ma2 + vb2 * 2 * ma2 * dma2) * pPrec2 * pPrec2;
            double pMeanTimesPrec2 = ma2 * mb2 * pPrec2;
            double pMeanTimesPrec2Deriv = dma2 * mb2 * pPrec2 + ma2 * dmb2 * pPrec2 + ma2 * mb2 * dpPrec2;
            return Gaussian.FromNatural(pMeanTimesPrec2Deriv - 1, 0);
        }

        [Skip]
        public static Gaussian ProductDeriv(Gaussian Product, [SkipIfUniform, Stochastic] Gaussian A, double B, Gaussian to_A)
        {
            return Gaussian.Uniform();
        }

        [Skip]
        public static Gaussian ProductDeriv(Gaussian Product, double A, [SkipIfUniform, Stochastic] Gaussian B, Gaussian to_B)
        {
            return ProductDeriv(Product, B, A, to_B);
        }

        /// <summary>VMP message to <c>product</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[sum_(b) p(b) factor(product,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageLogarithm(double A, [SkipIfUniform] Gaussian B)
        {
            if (B.IsPointMass)
                return Gaussian.PointMass(A * B.Point);
            if (A == 0)
                return Gaussian.PointMass(A);
            // m = A*mb
            // v = A*A*vb
            // 1/v = (1/vb)/(A*A)
            // m/v = (mb/vb)/A
            return Gaussian.FromNatural(B.MeanTimesPrecision / A, B.Precision / (A * A));
        }

        /// <summary>VMP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>product</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[sum_(a) p(a) factor(product,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian ProductAverageLogarithm([SkipIfUniform] Gaussian A, double B)
        {
            return ProductAverageLogarithm(B, A);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>product</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_product p(product) factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian Product, [Proper] Gaussian B)
        {
            if (B.IsPointMass)
                return AAverageLogarithm(Product, B.Point);
            if (Product.IsPointMass)
                return AAverageLogarithm(Product.Point, B);
            double mb, vb;
            B.GetMeanAndVariance(out mb, out vb);
            // note this is exact if B is a point mass (vb=0).
            Gaussian result = new Gaussian();
            result.Precision = Product.Precision * (vb + mb * mb);
            result.MeanTimesPrecision = Product.MeanTimesPrecision * mb;
            return result;
        }


        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(double Product, [Proper] Gaussian B)
        {
            // Throw an exception rather than return a meaningless point mass.
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> with <c>product</c> integrated out. The formula is <c>sum_product p(product) factor(product,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian Product, double B)
        {
            if (Product.IsPointMass)
                return AAverageLogarithm(Product.Point, B);
            return GaussianProductOp.AAverageConditional(Product, B);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian AAverageLogarithm(double Product, double B)
        {
            return GaussianProductOp.AAverageConditional(Product, B);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>product</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_product p(product) factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian BAverageLogarithm([SkipIfUniform] Gaussian Product, [Proper] Gaussian A)
        {
            return AAverageLogarithm(Product, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gaussian BAverageLogarithm(double Product, [Proper] Gaussian A)
        {
            return AAverageLogarithm(Product, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>product</c> integrated out. The formula is <c>sum_product p(product) factor(product,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gaussian BAverageLogarithm([SkipIfUniform] Gaussian Product, double A)
        {
            return AAverageLogarithm(Product, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian BAverageLogarithm(double Product, double A)
        {
            return AAverageLogarithm(Product, A);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Ratio(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Ratio", typeof(double), typeof(double))]
    [Quality(QualityBand.Mature)]
    public static class GaussianRatioVmpOp
    {
        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(ratio,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        internal const string NotSupportedMessage = "Variational Message Passing does not support a Ratio factor with fixed output or random denominator.";

        /// <summary>VMP message to <c>ratio</c>.</summary>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[sum_(b) p(b) factor(ratio,a,b)]</c>.</para>
        /// </remarks>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [NotSupported(GaussianRatioVmpOp.NotSupportedMessage)]
        public static Gaussian RatioAverageLogarithm(Gaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>ratio</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>ratio</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[sum_(a) p(a) factor(ratio,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gaussian RatioAverageLogarithm([SkipIfUniform] Gaussian A, double B)
        {
            return GaussianProductOp.AAverageConditional(A, B);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(ratio,a,b)))</c>.</para>
        /// </remarks>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [NotSupported(GaussianRatioVmpOp.NotSupportedMessage)]
        public static Gaussian AAverageLogarithm(Gaussian B)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="ratio">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> with <c>ratio</c> integrated out. The formula is <c>sum_ratio p(ratio) factor(ratio,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="ratio" /> is not a proper distribution.</exception>
        public static Gaussian AAverageLogarithm([SkipIfUniform] Gaussian ratio, double B)
        {
            return GaussianProductOp.ProductAverageConditional(ratio, B);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="ratio">Constant value for <c>ratio</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>The outgoing VMP message to the <c>a</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian AAverageLogarithm(double ratio, double B)
        {
            return GaussianProductOp.ProductAverageConditional(ratio, B);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        /// <remarks><para>
        /// Variational Message Passing does not support a Ratio factor with fixed output or random denominator
        /// </para></remarks>
        [NotSupported(GaussianRatioVmpOp.NotSupportedMessage)]
        public static Gaussian BAverageLogarithm()
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Ratio(double, double)" /></description></item><item><description><see cref="Factor.Product(double, double)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(double), typeof(double))]
    [FactorMethod(new string[] { "A", "Product", "B" }, typeof(Factor), "Ratio", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class WrappedGaussianProductOp
    {
        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[p(ratio) sum_(a) p(a) factor(ratio,a,b)]/p(ratio)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static WrappedGaussian AAverageConditional([SkipIfUniform] WrappedGaussian Product, double B, WrappedGaussian result)
        {
            result.Period = Product.Period / B;
            result.Gaussian = GaussianProductOp.AAverageConditional(Product.Gaussian, B);
            result.Normalize();
            return result;
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>ratio</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(a) p(a) factor(ratio,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static WrappedGaussian BAverageConditional([SkipIfUniform] WrappedGaussian Product, double A, WrappedGaussian result)
        {
            return AAverageConditional(Product, A, result);
        }

        /// <summary>EP message to <c>ratio</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>ratio</c> conditioned on the given values.</para>
        /// </remarks>
        public static WrappedGaussian AAverageConditional(double Product, double B, WrappedGaussian result)
        {
            if (B == 0)
            {
                if (Product != 0)
                    throw new AllZeroException();
                result.SetToUniform();
            }
            else
                result.Point = Product / B;
            result.Normalize();
            return result;
        }

        // ----------------------------------------------------------------------------------------------------------------------
        // VMP
        // ----------------------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="product">Incoming message from <c>a</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(WrappedGaussian product)
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="A">Incoming message from <c>ratio</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> with <c>ratio</c> integrated out. The formula is <c>sum_ratio p(ratio) factor(ratio,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static WrappedGaussian ProductAverageLogarithm([SkipIfUniform] WrappedGaussian A, double B, WrappedGaussian result)
        {
            double m, v;
            A.Gaussian.GetMeanAndVariance(out m, out v);
            result.Gaussian.SetMeanAndVariance(B * m, B * B * v);
            double period = B * A.Period;
            if (period != result.Period)
            {
                double ratio = period / result.Period;
                double intRatio = Math.Round(ratio);
                if (Math.Abs(ratio - intRatio) > result.Period * 1e-4)
                    throw new ArgumentException("B*A.Period (" + period + ") is not a multiple of result.Period (" + result.Period + ")");
                // if period is a multiple of result.Period, then wrapping to result.Period is equivalent to first wrapping to period, then to result.Period.
            }
            result.Normalize();
            return result;
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="A">Constant value for <c>ratio</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(ratio,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static WrappedGaussian ProductAverageLogarithm(double A, [SkipIfUniform] WrappedGaussian B, WrappedGaussian result)
        {
            return ProductAverageLogarithm(B, A, result);
        }

        /// <summary>VMP message to <c>ratio</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>ratio</c> as the random arguments are varied. The formula is <c>proj[sum_(a) p(a) factor(ratio,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static WrappedGaussian AAverageLogarithm([SkipIfUniform] WrappedGaussian Product, double B, WrappedGaussian result)
        {
            if (Product.IsPointMass)
                return AAverageLogarithm(Product.Point, B, result);
            return AAverageConditional(Product, B, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>a</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>ratio</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(ratio,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static WrappedGaussian BAverageLogarithm([SkipIfUniform] WrappedGaussian Product, double A, WrappedGaussian result)
        {
            return AAverageLogarithm(Product, A, result);
        }

        /// <summary>VMP message to <c>ratio</c>.</summary>
        /// <param name="Product">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>ratio</c> conditioned on the given values.</para>
        /// </remarks>
        public static WrappedGaussian AAverageLogarithm(double Product, double B, WrappedGaussian result)
        {
            return AAverageConditional(Product, B, result);
        }
    }
}
