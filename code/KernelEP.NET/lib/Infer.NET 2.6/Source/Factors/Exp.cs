// (C) Copyright 2009-2010 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double), Default = true)]
    [Quality(QualityBand.Stable)]
    public class ExpOp
    {
        /// <summary>
        /// The number of quadrature nodes used to compute the messages.
        /// Reduce this number to save time in exchange for less accuracy.
        /// </summary>
        public static int QuadratureNodeCount = 21;

        /// <summary>
        /// Number of quadrature iterations
        /// </summary>
        public static int QuadratureIterations = 2;

        /// <summary>
        /// Quadrature shift
        /// </summary>
        public static bool QuadratureShift = false; // gives a bit more accuracy when to_d is uniform.

        /// <summary>
        ///  Forces proper messages when set to true. 
        /// </summary>
        public static bool ForceProper = true;

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(exp,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double exp, double d)
        {
            return (exp == Math.Exp(d)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(exp,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double exp, double d)
        {
            return LogAverageFactor(exp, d);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(exp,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(double exp, double d)
        {
            return LogAverageFactor(exp, d);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp) p(exp) factor(exp,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gamma exp, double d)
        {
            return exp.GetLogProb(Math.Exp(d));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(d) p(d) factor(exp,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double exp, Gaussian d)
        {
            double log = Math.Log(exp);
            return d.GetLogProb(log) - log;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gamma exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return LogAverageFactor(exp, d.Point);
            if (d.IsUniform())
                return exp.GetLogAverageOf(new Gamma(0, 0));
            if (exp.IsPointMass)
                return LogAverageFactor(exp.Point, d);
            if (exp.IsUniform())
                return 0.0;
            double[] nodes = new double[QuadratureNodeCount];
            double[] weights = new double[QuadratureNodeCount];
            double mD, vD;
            Gaussian dMarginal = d * to_d;
            dMarginal.GetMeanAndVariance(out mD, out vD);
            Quadrature.GaussianNodesAndWeights(mD, vD, nodes, weights);
            if (!to_d.IsUniform())
            {
                // modify the weights to include q(y_k)/N(y_k;m,v)
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] *= Math.Exp(d.GetLogProb(nodes[i]) - Gaussian.GetLogProb(nodes[i], mD, vD));
                }
            }
            double Z = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                double y = nodes[i];
                double f = weights[i] * Math.Exp((exp.Shape - 1) * y - exp.Rate * Math.Exp(y));
                Z += f;
            }
            return Math.Log(Z) - exp.GetLogNormalizer();
        }

        /// <summary>
        /// Evidence message for EP
        /// </summary>
        /// <param name="exp">Incoming message from 'exp'.</param>
        /// <param name="d">Incoming message from 'd'.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions</returns>
        /// <remarks><para>
        /// The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d))</c>.
        /// </para></remarks>
        public static double LogAverageFactor_slow(Gamma exp, Gaussian d)
        {
            Gaussian to_d = Gaussian.Uniform();
            for (int i = 0; i < QuadratureIterations; i++)
            {
                to_d = DAverageConditional(exp, d, to_d);
            }
            return LogAverageFactor(exp, d, to_d);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_exp">Outgoing message to <c>exp</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d) / sum_exp p(exp) messageTo(exp))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Gamma exp, Gaussian d, [Fresh] Gamma to_exp, Gaussian to_d)
        {
            //Gaussian to_d = Gaussian.Uniform();
            //for (int i = 0; i < QuadratureIterations; i++) {
            //  to_d = DAverageConditional(exp, d, to_d);
            //}
            //Gamma to_exp = ExpAverageConditional(exp, d, to_d);
            return LogAverageFactor(exp, d, to_d)
                                - to_exp.GetLogAverageOf(exp);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(d) p(d) factor(exp,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double exp, Gaussian d)
        {
            return LogAverageFactor(exp, d);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gamma ExpAverageConditional(Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (d.IsUniform())
                return Gamma.FromShapeAndRate(0, 0);
            if (exp.IsPointMass)
            {
                // Z = int_y delta(x - exp(y)) N(y; my, vy) dy 
                //   = int_u delta(x - u) N(log(u); my, vy)/u du
                //   = N(log(x); my, vy)/x
                // logZ = -log(x) -0.5/vy*(log(x)-my)^2
                // dlogZ/dx = -1/x -1/vy*(log(x)-my)/x
                // d2logZ/dx2 = -dlogZ/dx/x -1/vy/x^2
                // log Ga(x;a,b) = (a-1)*log(x) - bx
                // dlogGa/dx = (a-1)/x - b
                // d2logGa/dx2 = -(a-1)/x^2
                // match derivatives and solve for (a,b)
                double shape = (1 + d.GetMean() - Math.Log(exp.Point)) * d.Precision;
                double rate = d.Precision / exp.Point;
                return Gamma.FromShapeAndRate(shape, rate);
            }
            if (exp.IsUniform())
                return ExpAverageLogarithm(d);

            if (to_d.IsUniform() && exp.Shape > 1)
            {
                to_d = new Gaussian(MMath.Digamma(exp.Shape - 1) - Math.Log(exp.Rate), MMath.Trigamma(exp.Shape - 1));
            }

            double mD, vD;
            Gaussian dMarginal = d * to_d;
            dMarginal.GetMeanAndVariance(out mD, out vD);
            double Z = 0;
            double sumy = 0;
            double sumexpy = 0;

            if (vD < 1e-6)
            {
                double m, v;
                d.GetMeanAndVariance(out m, out v);
                return Gamma.FromLogMeanAndMeanLog(m + v / 2.0, m);
            }

            //if (vD < 10)
            if (true)
            {
                // Use Gauss-Hermite quadrature
                double[] nodes = new double[QuadratureNodeCount];
                double[] weights = new double[QuadratureNodeCount];

                Quadrature.GaussianNodesAndWeights(mD, vD, nodes, weights);
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] = Math.Log(weights[i]);
                }
                if (!to_d.IsUniform())
                {
                    // modify the weights to include q(y_k)/N(y_k;m,v)
                    for (int i = 0; i < weights.Length; i++)
                    {
                        weights[i] += d.GetLogProb(nodes[i]) - dMarginal.GetLogProb(nodes[i]);
                    }
                }

                double maxLogF = Double.NegativeInfinity;
                // f(x,y) = Ga(exp(y); shape, rate) = exp(y*(shape-1) -rate*exp(y))
                // Z E[x] = int_y int_x x Ga(x;a,b) delta(x - exp(y)) N(y;my,vy) dx dy
                //        = int_y exp(y) Ga(exp(y);a,b) N(y;my,vy) dy
                // Z E[log(x)] = int_y y Ga(exp(y);a,b) N(y;my,vy) dy
                for (int i = 0; i < weights.Length; i++)
                {
                    double y = nodes[i];
                    double logf = weights[i] + (exp.Shape - 1) * y - exp.Rate * Math.Exp(y);
                    if (logf > maxLogF)
                        maxLogF = logf;
                    weights[i] = logf;
                }
                for (int i = 0; i < weights.Length; i++)
                {
                    double y = nodes[i];
                    double f = Math.Exp(weights[i] - maxLogF);
                    double f_y = f * y;
                    double fexpy = f * Math.Exp(y);
                    Z += f;
                    sumy += f_y;
                    sumexpy += fexpy;
                }
            }
            else
            {
                Converter<double, double> p = delegate(double y) { return d.GetLogProb(y) + (exp.Shape - 1) * y - exp.Rate * Math.Exp(y); };
                double sc = Math.Sqrt(vD);
                double offset = p(mD);
                Z = Quadrature.AdaptiveClenshawCurtis(z => Math.Exp(p(sc * z + mD) - offset), 1, 16, 1e-6);
                sumy = Quadrature.AdaptiveClenshawCurtis(z => (sc * z + mD) * Math.Exp(p(sc * z + mD) - offset), 1, 16, 1e-6);
                sumexpy = Quadrature.AdaptiveClenshawCurtis(z => Math.Exp(sc * z + mD + p(sc * z + mD) - offset), 1, 16, 1e-6);
            }
            if (Z == 0)
                throw new ApplicationException("Z==0");
            double s = 1.0 / Z;
            if (Double.IsPositiveInfinity(s))
                throw new ApplicationException("s is -inf");
            double meanLog = sumy * s;
            double mean = sumexpy * s;
            Gamma result = Gamma.FromMeanAndMeanLog(mean, meanLog);
            result.SetToRatio(result, exp, ForceProper);
            if (Double.IsNaN(result.Shape) || Double.IsNaN(result.Rate))
                throw new ApplicationException("result is nan");
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary />
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gamma ExpAverageConditionalInit([IgnoreDependency] Gaussian d)
        {
            return Gamma.Uniform();
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian DAverageConditional(double exp)
        {
            return Gaussian.PointMass(Math.Log(exp));
        }

        //internal static Gaussian DAverageConditional_slow([SkipIfUniform] Gamma exp, [Proper] Gaussian d)
        //{
        //  Gaussian to_d = exp.Shape<=1 || exp.Rate==0 ? 
        //            Gaussian.Uniform() 
        //            : new Gaussian(MMath.Digamma(exp.Shape-1) - Math.Log(exp.Rate), MMath.Trigamma(exp.Shape));
        //  //var to_d = Gaussian.Uniform();
        //  for (int i = 0; i < QuadratureIterations; i++) {
        //    to_d = DAverageConditional(exp, d, to_d);
        //  }
        //  return to_d;
        //}
        // to_d does not need to be Fresh. it is only used for quadrature proposal.

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian result)
        {
            if (exp.IsUniform() || d.IsPointMass)
                return Gaussian.Uniform();
            if (exp.IsPointMass)
                return DAverageConditional(exp.Point);
            if (exp.Rate < 0)
                throw new ImproperMessageException(exp);
            if (d.IsUniform())
            {
                if (exp.Shape <= 1)
                    throw new ArgumentException("The posterior has infinite variance due to input of Exp distributed as " + d + " and output of Exp distributed as " + exp +
                                                " (shape <= 1)");
                // posterior for d is a shifted log-Gamma distribution:
                // exp((a-1)*d - b*exp(d)) =propto exp(a*(d+log(b)) - exp(d+log(b)))
                // we find the Gaussian with same moments.
                // u = d+log(b)
                // E[u] = digamma(a-1)
                // E[d] = E[u]-log(b) = digamma(a-1)-log(b)
                // var(d) = var(u) = trigamma(a-1)
                double lnRate = Math.Log(exp.Rate);
                return new Gaussian(MMath.Digamma(exp.Shape - 1) - lnRate, MMath.Trigamma(exp.Shape - 1));
            }
            // We use moment matching to find the best Gaussian message.
            // The moments are computed via quadrature.
            // Z = int_y f(x,y) q(y) dy =approx sum_k w_k f(x,y_k) q(y_k)/N(y_k;m,v) 
            // f(x,y) = Ga(exp(y); shape, rate) = exp(y*(shape-1) -rate*exp(y))
            double[] nodes = new double[QuadratureNodeCount];
            double[] weights = new double[QuadratureNodeCount];
            double moD, voD;
            d.GetMeanAndVariance(out moD, out voD);
            double mD, vD;
            if (result.IsUniform() && exp.Shape > 1)
                result = new Gaussian(MMath.Digamma(exp.Shape - 1) - Math.Log(exp.Rate), MMath.Trigamma(exp.Shape - 1));
            Gaussian dMarginal = d * result;
            dMarginal.GetMeanAndVariance(out mD, out vD);
            Quadrature.GaussianNodesAndWeights(mD, vD, nodes, weights);
            if (!result.IsUniform())
            {
                // modify the weights to include q(y_k)/N(y_k;m,v)
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] *= Math.Exp(d.GetLogProb(nodes[i]) - Gaussian.GetLogProb(nodes[i], mD, vD));
                }
            }
            double Z = 0;
            double sumy = 0;
            double sumy2 = 0;
            double maxLogF = Double.NegativeInfinity;
            for (int i = 0; i < weights.Length; i++)
            {
                double y = nodes[i];
                double logf = Math.Log(weights[i]) + (exp.Shape - 1) * y - exp.Rate * Math.Exp(y);
                if (logf > maxLogF)
                    maxLogF = logf;
                weights[i] = logf;
            }
            for (int i = 0; i < weights.Length; i++)
            {
                double y = nodes[i];
                double f = Math.Exp(weights[i] - maxLogF);
                double f_y = f * y;
                double fyy = f_y * y;
                Z += f;
                sumy += f_y;
                sumy2 += fyy;
            }
            if (Z == 0)
                return Gaussian.Uniform();
            double s = 1.0 / Z;
            double mean = sumy * s;
            double var = sumy2 * s - mean * mean;
            // TODO: explain this
            if (var <= 0.0)
            {
                double quadratureGap = 0.1;
                var = 2 * vD * quadratureGap * quadratureGap;
            }
            result = new Gaussian(mean, var);
            result.SetToRatio(result, d, ForceProper);
            if (result.Precision < -1e10)
                throw new ApplicationException("result has negative precision");
            if (Double.IsPositiveInfinity(result.Precision))
                throw new ApplicationException("result is point mass");
            if (Double.IsNaN(result.Precision) || Double.IsNaN(result.MeanTimesPrecision))
                throw new ApplicationException("result is nan");
            return result;
        }

        //-- VMP -------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(exp,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor( /*[Proper] Gamma exp, [Proper] Gaussian d*/)
        {
            //double mean, variance;
            //d.GetMeanAndVariance(out mean, out variance);
            //return (exp.Shape - 1) * mean - exp.Rate * Math.Exp(mean + variance / 2)
            //    - MMath.GammaLn(exp.Shape) + exp.Shape * Math.Log(exp.Rate);
            return 0.0;
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gamma ExpAverageLogarithmInit()
        {
            return Gamma.Uniform();
        }

        /// <summary>VMP message to <c>exp</c>.</summary>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[sum_(d) p(d) factor(exp,d)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gamma ExpAverageLogarithm([Proper] Gaussian d)
        {
            double mD, vD;
            d.GetMeanAndVariance(out mD, out vD);
            double lm = mD + vD / 2;
            //return Gamma.FromMeanAndVariance(Math.Exp(lm), Math.Exp(2*lm)*(Math.Exp(vD)-1));
            return Gamma.FromLogMeanAndMeanLog(lm, mD);
        }

        /// <summary>VMP message to <c>exp</c>.</summary>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[sum_(d) p(d) factor(exp,d)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static Gamma ExpAverageLogarithm([Proper] NonconjugateGaussian d)
        {
            double mD, vD;
            d.GetMeanAndVariance(out mD, out vD);
            double lm = mD + vD / 2;
            //return Gamma.FromMeanAndVariance(Math.Exp(lm), Math.Exp(2*lm)*(Math.Exp(vD)-1));
            return Gamma.FromLogMeanAndMeanLog(lm, mD);
        }

        // Finds the maximum of -C exp(v/2) + .5 log(v)
        // Should converge in <=10 iterations
        private static double FindMinimumInV(double C)
        {
            double THRESHOLD = 1E-3;
            int MAX_ITS = 30;
            double v = -Math.Log(C) + 0.31;
            if (C > 0.2178)
            {
                double old_v = v;
                for (int i = 0; i < MAX_ITS; i++)
                {
                    v = Math.Exp(-v / 2.0) / C;
                    if (Math.Abs(old_v - v) < THRESHOLD)
                        return v;
                    old_v = v;
                }
            }
            else if (C < 0.1576)
            {
                double old_v = v;
                for (int i = 0; i < MAX_ITS; i++)
                {
                    v = -2.0 * Math.Log(v * C);
                    if (v < 0)
                        throw new ApplicationException("FindMinimumInV failed for C= " + C);
                    if (Math.Abs(old_v - v) < THRESHOLD)
                        return v;
                    old_v = v;
                }
            }
            return v;
        }

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> with <c>exp</c> integrated out. The formula is <c>sum_exp p(exp) factor(exp,d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        [Quality(QualityBand.Experimental)]
        public static NonconjugateGaussian DAverageLogarithm([Proper] Gamma exp, [Proper, SkipIfUniform] NonconjugateGaussian d, NonconjugateGaussian result)
        {
            return DNonconjugateAverageLogarithm(exp, d.GetGaussian(true), result);
        }

        /// <summary>
        /// Nonconjugate VMP message to 'd'.
        /// </summary>
        /// <param name="exp">Incoming message from 'exp'. Must be a proper distribution.  If uniform, the result will be uniform.</param>
        /// <param name="d"></param>
        /// <param name="result"></param>
        /// <returns>The outgoing nonconjugate VMP message to the 'd' argument.</returns>
        /// <remarks><para>
        /// The outgoing message is the exponential of the integral of the log-factor times incoming messages, over all arguments except 'd'.
        /// The formula is <c>int log(f(d,x)) q(x) dx</c> where <c>x = (exp)</c>.
        /// </para></remarks>
        /// <exception cref="ImproperMessageException"><paramref name="exp"/> is not a proper distribution</exception>
        [Quality(QualityBand.Experimental)]
        public static NonconjugateGaussian DNonconjugateAverageLogarithm([Proper] Gamma exp, [Proper, SkipIfUniform] Gaussian d, NonconjugateGaussian result)
        {
            var vf = -1.0;
            var a = exp.Shape;
            var v_opt = 1.0 / (a - 1.0);
            var b = exp.Rate;
            double m = 0.0, v = 0.0;
            d.GetMeanAndVariance(out m, out v);
            if (a > 1)
            {
                var mf = Math.Log((a - 1) / b) - .5 * v_opt;
                if (mf != m)
                {
                    var grad_S_m = -b * Math.Exp(m + v / 2) + a - 1;
                    vf = (mf - m) / grad_S_m;
                    result.MeanTimesPrecision = mf / vf;
                    result.Precision = 1 / vf;
                }
            }
            //m = (mp + prior_mp)/(p + prior_p);
            if (vf < 0)
            {
                result.Precision = b * Math.Exp(m + v / 2);
                result.MeanTimesPrecision = (m - 1) * result.Precision + a - 1;
            }

            double bf = -1, afm1 = -1;
            if (a <= 1)
                v_opt = FindMinimumInV(b * Math.Exp(m));
            if (v_opt != v)
            {
                var grad_S_v = -.5 * b * Math.Exp(m + v / 2);
                bf = v * grad_S_v / (v_opt - v);
                afm1 = v_opt * bf;
            }
            if (afm1 < 0 || bf < 0)
            {
                afm1 = b * v * v * Math.Exp(m + v / 2) / 4;
                bf = b * (1 + v / 2) * Math.Exp(m + v / 2) / 2;
            }
            result.Shape = afm1 + 1;
            result.Rate = bf;
            if (!result.IsProper())
                throw new ApplicationException("improper message calculated by ExpOp.DNonconjugateAverageLogarithm");
            return result;
        }

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing VMP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> with <c>exp</c> integrated out. The formula is <c>sum_exp p(exp) factor(exp,d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageLogarithm([Proper] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            if (exp.IsPointMass)
                return DAverageLogarithm(exp.Point);

            double m, v;
            d.GetMeanAndVariance(out m, out v);
            /* --------- This update comes from T. Minka's equations for non-conjugate VMP. -------- */
            Gaussian result = new Gaussian();
            result.Precision = exp.Rate * Math.Exp(m + v / 2);
            result.MeanTimesPrecision = (m - 1) * result.Precision + (exp.Shape - 1);
            double step = Rand.Double() * 0.5; // random damping helps convergence, especially with parallel updates
            if (step != 1.0)
            {
                result.Precision = step * result.Precision + (1 - step) * to_d.Precision;
                result.MeanTimesPrecision = step * result.MeanTimesPrecision + (1 - step) * to_d.MeanTimesPrecision;
            }
            if (result.IsPointMass)
                throw new Exception();
            return result;
        }

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="exp">Constant value for <c>exp</c>.</param>
        /// <returns>The outgoing VMP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian DAverageLogarithm(double exp)
        {
            return DAverageConditional(exp);
        }

        private const string NotSupportedMessage = "VMP cannot support deterministic factors (e.g. Exp) with fixed output";

        /// <summary>VMP message to <c>exp</c>.</summary>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>The outgoing VMP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>exp</c> conditioned on the given values.</para>
        /// </remarks>
        [NotSupported(NotSupportedMessage)]
        public static Gamma ExpAverageLogarithm(double d)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp_Laplace
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            Gaussian dPost = d * to_d;
            double x = dPost.GetMean();
            double v = dPost.GetVariance();
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            return exp.GetLogProb(expx) + d.GetLogProb(x) + MMath.LnSqrt2PI + 0.5 * Math.Log(v);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="to_exp">Previous outgoing message to <c>exp</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d) / sum_exp p(exp) messageTo(exp))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, Gamma to_exp)
        {
            return LogAverageFactor(exp, d, to_d) - to_exp.GetLogAverageOf(exp);
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf = (a - 1) - b * ehat;
            double ddlogf = -b * ehat;
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * dhat + dlogf, r);
        }

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf_diff = -ehat;
            double ddlogf_diff = 0;
            double ddlogfx = -ehat;
            double dlogz_diff = dlogf_diff;
            double dx = ddlogfx / (d.Precision - ddlogf_diff + a / (b * b));
            double ddlogz_diff = ddlogf_diff + ddlogfx * dx;
            double m = -dlogz_diff;
            double v = ddlogz_diff;
            Gamma result = Gamma.FromMeanAndVariance(m, v);
            result.SetToRatio(result, exp, true);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Buffers("x")]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp_Laplace2
    {
        /// <summary>Initialize the buffer <c>x</c>.</summary>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <returns>Initial value of buffer <c>x</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double XInit(Gaussian d)
        {
            return d.GetMean();
        }

        /// <summary>Update the buffer <c>x</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="x">Buffer <c>x</c>.</param>
        /// <returns>New value of buffer <c>x</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double X([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x)
        {
            // perform one Newton update of X
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double r = b * expx;
            double dlogf = (a - 1) - r;
            //if (r < 0) r = 0;
            double t = r * x + dlogf;
            return (t + d.MeanTimesPrecision) / (r + d.Precision);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="x">Buffer <c>x</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x)
        {
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double v = 1 / (d.Precision + b * expx);
            return exp.GetLogProb(expx) + d.GetLogProb(x) + MMath.LnSqrt2PI + 0.5 * Math.Log(v);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="x">Buffer <c>x</c>.</param>
        /// <param name="to_exp">Previous outgoing message to <c>exp</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d) / sum_exp p(exp) messageTo(exp))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x, Gamma to_exp)
        {
            return LogAverageFactor(exp, d, x) - to_exp.GetLogAverageOf(exp);
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="x">Buffer <c>x</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double x)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double ddlogf = -b * expx;
            double dddlogf = ddlogf;
            double d4logf = ddlogf;
            double dlogf = (a - 1) + ddlogf;
            double v = 1 / (d.Precision - ddlogf);
            // this is Laplace's estimate of the posterior mean and variance
            double mpost = x + 0.5 * dddlogf * v * v;
            double vx = d.GetVariance();
            double vpost = vx * (1 + v * (ddlogf + 0.5 * d4logf * v * v * d.Precision + dddlogf * dddlogf * v * v * v * d.Precision));
            Gaussian result = Gaussian.FromMeanAndVariance(mpost, vpost);
            result.SetToRatio(result, d, true);
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="x">Buffer <c>x</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, double x)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            double a = exp.Shape;
            double b = exp.Rate;
            double expx = Math.Exp(x);
            double ddlogf = -b * expx;
            double v = 1 / (d.Precision - ddlogf);
            double logz = LogAverageFactor(exp, d, x) + exp.GetLogNormalizer();
            Gamma exp1 = Gamma.FromShapeAndRate(a + 1, b);
            double x1 = x;
            for (int iter = 0; iter < 100; iter++)
            {
                double oldx1 = x1;
                x1 = X(exp1, d, x1);
                if (Math.Abs(x1 - oldx1) < 1e-10)
                    break;
            }
            double logz1 = LogAverageFactor(exp1, d, x1) + exp1.GetLogNormalizer();
            Gamma exp2 = Gamma.FromShapeAndRate(a + 2, b);
            double x2 = x;
            for (int iter = 0; iter < 100; iter++)
            {
                double oldx2 = x2;
                x2 = X(exp2, d, x2);
                if (Math.Abs(x2 - oldx2) < 1e-10)
                    break;
            }
            double logz2 = LogAverageFactor(exp2, d, x2) + exp2.GetLogNormalizer();
            double mpost = Math.Exp(logz1 - logz);
            double m2post = Math.Exp(logz2 - logz);
            double vpost = m2post - mpost * mpost;
            double dddlogf = ddlogf;
            if (false)
            {
                // this is better for large b, worse for small b
                double mpost2 = expx + 0.5 * v * expx + 0.5 * v * v * expx * dddlogf;
                Console.WriteLine(mpost2);
            }
            if (false)
            {
                // this is identical to above
                double dlogf = (a - 1) + ddlogf;
                double dlogz = dlogf + 0.5 * v * v * dddlogf * d.Precision;
                double mpost2 = ((a - 1) - dlogz) / b; // expx+0.5*v*v*prec*expx
                Console.WriteLine(mpost2);
            }
            Gamma result = Gamma.FromMeanAndVariance(mpost, vpost);
            result.SetToRatio(result, exp, true);
            return result;
        }
    }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Buffers("c")]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp_Laplace3
    {
        public static bool modified = true;

        /// <summary>Initialize the buffer <c>c</c>.</summary>
        /// <returns>Initial value of buffer <c>c</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static double CInit()
        {
            return 0.0;
        }

        /// <summary>Update the buffer <c>c</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>New value of buffer <c>c</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double C([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
        {
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogh = b * ehat / (d.Precision + c + b * ehat);
            double ddlogh = dlogh * (1 - dlogh);
            return 0.5 * ddlogh;
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf = (a - 1) - b * ehat;
            double ddlogf = -b * ehat;
            if (modified)
            {
                double h = d.Precision + c - ddlogf;
                double dlogh = b * ehat / h;
                double ddlogh = dlogh * (1 - dlogh);
                dlogf -= 0.5 * dlogh;
                ddlogf -= 0.5 * ddlogh;
            }
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * dhat + dlogf, r);
        }

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d, double c)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf_diff = -ehat;
            double ddlogf_diff = 0;
            double ddlogfx = -ehat;
            if (modified)
            {
                double dlogh = ehat / (d.Precision + c + b * ehat);
                double ddlogh = -dlogh * dlogh;
                double ddloghx = dlogh * (1 - b * dlogh);
                dlogf_diff -= 0.5 * dlogh;
                ddlogf_diff -= 0.5 * ddlogh;
                ddlogfx -= 0.5 * ddloghx;
            }
            double dlogz_diff = dlogf_diff;
            double dx = ddlogfx / (d.Precision - ddlogf_diff + a / (b * b));
            double ddlogz_diff = ddlogf_diff + ddlogfx * dx;
            double m = -dlogz_diff;
            double v = ddlogz_diff;
            Gamma result = Gamma.FromMeanAndVariance(m, v);
            result.SetToRatio(result, exp, true);
            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d)
        {
            Gaussian dPost = d * to_d;
            double x = dPost.GetMean();
            double expx = Math.Exp(x);
            double a = exp.Shape;
            double b = exp.Rate;
            double v = dPost.GetVariance();
            return exp.GetLogProb(expx) + d.GetLogProb(x) + MMath.LnSqrt2PI + 0.5 * Math.Log(v);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="to_exp">Previous outgoing message to <c>exp</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(exp,d) p(exp,d) factor(exp,d) / sum_exp p(exp) messageTo(exp))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double LogEvidenceRatio([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, Gamma to_exp)
        {
            return LogAverageFactor(exp, d, to_d) - to_exp.GetLogAverageOf(exp);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Buffers("logh")]
    [Quality(QualityBand.Experimental)]
    internal static class ExpOp_Laplace3_Slow
    {
        public static bool modified = true;
        public const int n = 100;
        public const double xmin = -10;
        public const double xmax = 5;
        private const double inc = (xmax - xmin) / (n - 1);
        private static Matrix dd = null;

        private static DenseVector Aux([SkipIfUniform] Gamma exp, [Proper] Gaussian d, DenseVector logh, DenseVector aux)
        {
            double b = exp.Rate;
            double dPrec = d.Precision;
            DenseVector target = DenseVector.Zero(n);
            for (int i = 0; i < n; i++)
            {
                double x = xmin + i * inc;
                target[i] = dPrec + b * Math.Exp(x) + 0.5 * aux[i];
            }
            DenseVector numer = DenseVector.Zero(n);
            numer.SetToProduct(dd, logh);
            for (int i = 0; i < n; i++)
            {
                double c = 0.5 / target[i];
                if (target[i] < 0)
                    throw new Exception();
                double g = (logh[i] - Math.Log(target[i]) + c * aux[i]) * c;
                aux[i] = (numer[i] + g) / (1 + c * c);
            }
            return aux;
        }

        /// <summary>Initialize the buffer <c>logh</c>.</summary>
        /// <returns>Initial value of buffer <c>logh</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double[] LoghInit()
        {
            return new double[n];
        }

        /// <summary>Update the buffer <c>logh</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double[] Logh([SkipIfUniform] Gamma exp, [Proper] Gaussian d, double[] result)
        {
            if (dd == null)
                dd = Deriv2Matrix(n, 1 / (inc * inc));
            double b = exp.Rate;
            double dPrec = d.Precision;
            DenseVector aux = DenseVector.Zero(n);
            PositiveDefiniteMatrix denom = new PositiveDefiniteMatrix(n, n);
            denom.SetToOuterTranspose(dd);
            for (int i = 0; i < n; i++)
            {
                denom[i, i] += 1;
            }
            DenseVector logh = DenseVector.Zero(n);
            for (int i = 0; i < n; i++)
            {
                double x = xmin + i * inc;
                logh[i] = Math.Log(dPrec + b * Math.Exp(x));
            }
            for (int iter = 0; iter < 100; iter++)
            {
                if (iter == 0)
                    aux.SetToProduct(dd, logh);
                aux = Aux(exp, d, logh, aux);
                logh.SetToProduct(aux, dd);
                for (int i = 0; i < n; i++)
                {
                    double x = xmin + i * inc;
                    double target = dPrec + b * Math.Exp(x) + 0.5 * aux[i];
                    if (target < 0)
                        throw new Exception();
                    logh[i] += Math.Log(target);
                }
                logh.PredivideBy(denom);
            }
            for (int i = 0; i < n; i++)
            {
                result[i] = logh[i];
            }
            return result;
        }

        public static Matrix Deriv2Matrix(int n, double scale)
        {
            Matrix dd = new Matrix(n, n);
            for (int i = 1; i < (n - 1); i++)
            {
                dd[i, i] = -2 * scale;
                dd[i, i + 1] = 1 * scale;
                dd[i, i - 1] = 1 * scale;
            }
            for (int j = 0; j < n; j++)
            {
                dd[0, j] = 2 * dd[1, j] - dd[2, j];
                dd[n - 1, j] = 2 * dd[n - 2, j] - dd[n - 3, j];
            }
            return dd;
        }

        public static double Interpolate(double[] y, double x)
        {
            double ifrac = (x - xmin) / inc;
            int i = (int)Math.Floor(ifrac);
            // interpolate along the line joining (i*inc, y[i]) and ((i+1)*inc, y[i+1])
            //double slope = (y[i+1]-y[i])/inc;
            //return y[i] + slope*(ifrac - i)*inc;
            double w = (ifrac - i);
            return y[i + 1] * w + y[i] * (1 - w);
        }

        public static void GetDerivatives(double[] y, double x, out double deriv, out double deriv2)
        {
            double y0 = Interpolate(y, x);
            double yPlus = Interpolate(y, x + inc);
            double yMinus = Interpolate(y, x - inc);
            deriv = (yPlus - yMinus) / (2 * inc);
            deriv2 = (yPlus + yMinus - 2 * y0) / (inc * inc);
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="logh">Buffer <c>logh</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double[] logh)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf = (a - 1) - b * ehat;
            double ddlogf = -b * ehat;
            if (modified)
            {
                double dlogh, ddlogh;
                GetDerivatives(logh, dhat, out dlogh, out ddlogh);
                dlogf -= 0.5 * dlogh;
                ddlogf -= 0.5 * ddlogh;
            }
            double r = Math.Max(0, -ddlogf);
            return Gaussian.FromNatural(r * dhat + dlogf, r);
        }

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogf_diff = -ehat;
            double ddlogf_diff = 0;
            double ddlogfx = -ehat;
            if (modified)
            {
                double c = 0;
                double dlogh = ehat / (d.Precision + c + b * ehat);
                double ddlogh = -dlogh * dlogh;
                double ddloghx = dlogh * (1 - b * dlogh);
                dlogf_diff -= 0.5 * dlogh;
                ddlogf_diff -= 0.5 * ddlogh;
                ddlogfx -= 0.5 * ddloghx;
            }
            double dlogz_diff = dlogf_diff;
            double dx = ddlogfx / (d.Precision - ddlogf_diff + a / (b * b));
            double ddlogz_diff = ddlogf_diff + ddlogfx * dx;
            double m = -dlogz_diff;
            double v = ddlogz_diff;
            Gamma result = Gamma.FromMeanAndVariance(m, v);
            result.SetToRatio(result, exp, true);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Buffers("c")]
    [Quality(QualityBand.Experimental)]
    internal static class ExpOp_Laplace2_Slow
    {
        public static bool modified = true;

        /// <summary>Initialize the buffer <c>c</c>.</summary>
        /// <returns>Initial value of buffer <c>c</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static double CInit()
        {
            return 0.0;
        }

        /// <summary>Update the buffer <c>c</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>New value of buffer <c>c</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double C([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
        {
            //return 0.0;
            Gaussian dPost = d * to_d;
            double h = dPost.Precision;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double eh = Math.Exp(dhat + 0.5 / h);
            double gx = (a - 1) - b * eh;
            double gh = 0.5 / (h * h) * b * eh;
            double gxx = -b * eh;
            double gxh = gh;
            double ghh = -gh * (2 / h + 0.5 / (h * h));
            double gxxx = gxx;
            double gxxxx = gxx;
            h = d.Precision + c - gxx;
            double dh = -gxxx;
            double ddh = -gxxxx;
            double dlogh = dh / h;
            double ddlogh = ddh / h - dlogh * dlogh;
            double beta2 = 2 * gxh * dh + ghh * (dh * dh) + gh * ddh - 0.5 * ddlogh;
            return -beta2;
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            //c = 0;
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double h = dPost.Precision;
            for (int i = 0; i < 10; i++)
            {
                h = d.Precision + c + b * Math.Exp(dhat + 0.5 / h);
            }
            double eh = Math.Exp(dhat + 0.5 / h);
            double gx = (a - 1) - b * eh;
            double gh = 0.5 / (h * h) * b * eh;
            double gxx = -b * eh;
            double gxh = gh;
            double ghh = -gh * (2 / h + 0.5 / (h * h));
            double gxxx = gxx;
            double gxxxx = gxx;
            //h = d.Precision + c - gxx;
            Console.WriteLine("h = {0}, {1}", h, dPost.Precision);
            double dh = -gxxx;
            double ddh = -gxxxx;
            double dlogh = dh / h;
            double ddlogh = ddh / h - dlogh * dlogh;
            double dlogz = gx + gh * dh - 0.5 * dlogh;
            double beta = gxx + 2 * gxh * dh + ghh * (dh * dh) + gh * ddh - 0.5 * ddlogh;
            double r = Math.Max(0, -beta);
            return Gaussian.FromNatural(r * dhat + dlogz, r);
        }

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d, double c)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            Gaussian dPost = d * to_d;
            double dhat = dPost.GetMean();
            double ehat = Math.Exp(dhat);
            double a = exp.Shape;
            double b = exp.Rate;
            double h = dPost.Precision;
            double eh = Math.Exp(dhat + 0.5 / h);
            double gx = (a - 1) - b * eh;
            double gh = 0.5 / (h * h) * b * eh;
            double gxx = -b * eh;
            double gxh = gh;
            double ghh = -gh * (2 / h + 0.5 / (h * h));
            double gxxx = gxx;
            double gxxxx = gxx;
            //h = d.Precision + c - gxx;
            Console.WriteLine("h = {0}, {1}", h, dPost.Precision);
            double dh = -gxxx;
            double ddh = -gxxxx;
            double dlogh = dh / h;
            double ddlogh = ddh / h - dlogh * dlogh;
            double gb = a / b - eh;
            double gb_diff = -eh;
            double gxb = -eh;
            double ghb = 0.5 / (h * h) * eh;
            double gbb = -a / (b * b);
            double gxxb = gxb;
            double gxxbb = 0;
            double hb = -gxxb;
            double hbb = -gxxbb;
            double dhb = hb;
            double dloghb = hb / h;
            double ddloghbb = hbb / h - dloghb * dloghb;
            double ddloghb = dhb / h - dloghb * dlogh;
            double dlogz_diff = gb_diff + gh * hb - 0.5 * dloghb;
            double alpha = gxb + ghb * dh + gxh * hb + ghh * dh * hb + gh * dhb - 0.5 * ddloghb;
            //double ddlogz = gbb + 2*ghb*hb - 0.5*ddloghbb + alpha*alpha/h;
            double ddlogz_diff = 2 * ghb * hb - 0.5 * ddloghbb + alpha * alpha / h;
            //double dlogz_diff = dlogz - a/b;
            //double ddlogz_diff = ddlogz + a/(b*b);
            double m = -dlogz_diff;
            double v = ddlogz_diff;
            Gamma result = Gamma.FromMeanAndVariance(m, v);
            result.SetToRatio(result, exp, true);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Buffers("c")]
    [Quality(QualityBand.Experimental)]
    public static class ExpOp3
    {
        /// <summary>Initialize the buffer <c>c</c>.</summary>
        /// <returns>Initial value of buffer <c>c</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static double CInit()
        {
            return 0.0;
        }

        /// <summary>Update the buffer <c>c</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>New value of buffer <c>c</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static double C([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
        {
            Gaussian dPost = d * to_d;
            double m, v;
            dPost.GetMeanAndVariance(out m, out v);
            double a = exp.Shape;
            double b = exp.Rate;
            return -0.5 * b * b * Math.Exp(2 * m + v) / (1 / (v * v) + 0.5 * b * Math.Exp(m + v / 2));
            //return -0.5*b*b*Math.Exp(2*m+v)*v*v;
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <param name="c">Buffer <c>c</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(exp) p(exp) factor(exp,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageConditional([SkipIfUniform] Gamma exp, [Proper] Gaussian d, Gaussian to_d, double c)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageConditional(exp.Point);
            Gaussian dPost = d * to_d;
            double m, v;
            dPost.GetMeanAndVariance(out m, out v);
            double ee = Math.Exp(m + v / 2);
            //ee = Math.Exp(m)*(1 + v/2);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogz = (a - 1) - b * ee;
            double t = -b * ee - c;
            //double ddlogz = b*b*Math.Exp(2*m+v)*MMath.ExpMinus1(v) - b*ee;
            double ddlogz = t / (1 - t / d.Precision);
            //double t = d.Precision*ddlogz/(d.Precision + ddlogz);
            //Console.WriteLine("dPost = {0}, t = {1}, postvar = {2}", dPost, t, (1 + ddlogz/d.Precision)/d.Precision);
            double r = Math.Max(0, -t);
            return Gaussian.FromNatural(r * m + dlogz, r);
        }

        /// <summary>EP message to <c>exp</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing EP message to the <c>exp</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>exp</c> as the random arguments are varied. The formula is <c>proj[p(exp) sum_(d) p(d) factor(exp,d)]/p(exp)</c>.</para>
        /// </remarks>
        public static Gamma ExpAverageConditional(Gamma exp, Gaussian d, Gaussian to_d)
        {
            if (d.IsPointMass)
                return Gamma.PointMass(Math.Exp(d.Point));
            if (exp.IsPointMass)
                return Gamma.Uniform();
            Gaussian dPost = d * to_d;
            double m, v;
            dPost.GetMeanAndVariance(out m, out v);
            double ee = Math.Exp(m + v / 2);
            double a = exp.Shape;
            double b = exp.Rate;
            double dlogz_diff = -ee;
            double ddlogz_diff = Math.Exp(2 * m + v) * MMath.ExpMinus1(v);
            double me = -dlogz_diff;
            double ve = ddlogz_diff;
            Gamma result = Gamma.FromMeanAndVariance(me, ve);
            result.SetToRatio(result, exp, true);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Exp(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Exp", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class ExpOp_BFGS
    {
        private static double GradientAndValueAtPoint(double mu, double s2, Vector x, double a, double b, Vector grad)
        {
            double m, v;
            m = x[0];
            v = Math.Exp(x[1]);
            double kl_value = -.5 * (1.0 + x[1]) + .5 * (v + (m - mu) * (m - mu)) / s2 + .5 * Math.Log(s2)
                              - ((a - 1) * m - b * Math.Exp(m + v / 2.0)); // + const
            if (grad != null)
            {
                grad[0] = (m - mu) / s2 - ((a - 1) - b * Math.Exp(m + v / 2.0));
                grad[1] = -.5 + .5 * v / s2 + b * .5 * v * Math.Exp(m + v / 2.0);
            }
            return kl_value;
        }

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="exp">Incoming message from <c>exp</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_d">Previous outgoing message to <c>d</c>.</param>
        /// <returns>The outgoing VMP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> with <c>exp</c> integrated out. The formula is <c>sum_exp p(exp) factor(exp,d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="exp" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian DAverageLogarithm([Proper] Gamma exp, [Proper, Stochastic] Gaussian d, Gaussian to_d)
        {
            if (exp.IsPointMass)
                return ExpOp.DAverageLogarithm(exp.Point);

            double m, v;
            d.GetMeanAndVariance(out m, out v);
            double mu, s2;
            var prior = d / to_d;
            prior.GetMeanAndVariance(out mu, out s2);
            var z = Vector.Zero(2);
            z[0] = m;
            z[1] = Math.Log(v);
            double startingValue = GradientAndValueAtPoint(mu, s2, z, exp.Shape, exp.Rate, null);
            var s = new BFGS();
            int evalCounter = 0;
            s.MaximumStep = 1e3;
            s.MaximumIterations = 100;
            s.Epsilon = 1e-5;
            s.convergenceCriteria = BFGS.ConvergenceCriteria.Objective;
            z = s.Run(z, 1.0, delegate(Vector y, ref Vector grad)
                {
                    evalCounter++;
                    return GradientAndValueAtPoint(mu, s2, y, exp.Shape, exp.Rate, grad);
                });
            m = z[0];
            v = Math.Exp(z[1]);
            to_d.SetMeanAndVariance(m, v);
            to_d.SetToRatio(to_d, prior);
            double endValue = GradientAndValueAtPoint(mu, s2, z, exp.Shape, exp.Rate, null);
            //Console.WriteLine("Went from {0} to {1} in {2} steps, {3} evals", startingValue, endValue, s.IterationsPerformed, evalCounter);
            if (startingValue < endValue)
                Console.WriteLine("Warning: BFGS resulted in an increased objective function");
            return to_d;

            /* ---------------- NEWTON ITERATION VERSION 1 ------------------- 
            double meanTimesPrec, prec;
            d.GetNatural(out meanTimesPrec, out prec);
            Matrix K = new Matrix(2, 2);
            K[0, 0]=1/prec; // d2K by dmu^2
            K[1, 0]=K[0, 1]=-meanTimesPrec/(prec*prec);
            K[1, 1]=meanTimesPrec*meanTimesPrec/Math.Pow(prec, 3)+1/(2*prec*prec);
            double[,,] Kprime = new double[2, 2, 2];
            Kprime[0, 0, 0]=0;
            Kprime[0, 0, 1]=Kprime[0, 1, 0]=Kprime[1, 0, 0]=-1/(prec*prec);
            Kprime[0, 1, 1]=Kprime[1, 1, 0]=Kprime[1, 0, 1]=2*meanTimesPrec/Math.Pow(prec, 3);
            Kprime[1, 1, 1]=-3*meanTimesPrec*meanTimesPrec/Math.Pow(prec, 4)-1/Math.Pow(prec, 3);
            Vector gradS = new Vector(2);
            gradS[0]=(exp.Shape-1)/prec-exp.Rate/prec*Math.Exp((meanTimesPrec+.5)/prec);
            gradS[1]=-(exp.Shape-1)*meanTimesPrec/(prec*prec)+exp.Rate*(meanTimesPrec+.5)/(prec*prec)*Math.Exp((meanTimesPrec+.5)/prec);
            Matrix grad2S = new Matrix(2, 2);
            grad2S[0, 0]=-exp.Rate/(prec*prec)*Math.Exp((meanTimesPrec+.5)/prec);
            grad2S[0, 1]=grad2S[1, 0]=-(exp.Shape-1)/(prec*prec)+exp.Rate*(1/(prec*prec)+(meanTimesPrec+.5)/Math.Pow(prec, 3))*Math.Exp((meanTimesPrec+.5)/prec);
            grad2S[1, 1]=2*(exp.Shape-1)*meanTimesPrec/Math.Pow(prec, 3)-exp.Rate*(meanTimesPrec+.5)/(prec*prec)*(2/prec+(meanTimesPrec+.5)/(prec*prec))*Math.Exp((meanTimesPrec+.5)/prec);
            Vector phi = new Vector(new double[] { result.MeanTimesPrecision, result.Precision });
            Vector gradKL = K*phi-gradS;
            Matrix hessianKL = K - grad2S;
            for (int i=0; i<2; i++)
                            for (int j=0; j<2; j++)
                                            for (int k=0; k<2; k++)
                                                            hessianKL[i, j]+=Kprime[i, j, k]*phi[k];
            double step = 1;
            Vector direction = GammaFromShapeAndRate.twoByTwoInverse(hessianKL)*gradKL;
            Vector newPhi = phi - step * direction;
            result.SetNatural(newPhi[0], newPhi[1]);
            return result;

            ---------------- NEWTON ITERATION VERSION 2 ------------------- 
            double mean, variance;
            d.GetMeanAndVariance(out mean, out variance); 
            Gaussian prior = d / result; 
            double mean1, variance1;
            prior.GetMeanAndVariance(out mean1, out variance1); 
            Vector gradKL = new Vector(2);
            gradKL[0]=-(exp.Shape-1)+exp.Rate*Math.Exp(mean+variance/2)+mean/variance1-mean1/variance1;
            gradKL[1]=-1/(2*variance)+exp.Rate*Math.Exp(mean+variance/2)+1/(2*variance1);
            Matrix hessianKL = new Matrix(2, 2);
            hessianKL[0, 0]=exp.Rate*Math.Exp(mean+variance/2)+1/variance1;
            hessianKL[0, 1]=hessianKL[1, 0]=.5*exp.Rate*Math.Exp(mean+variance/2);
            hessianKL[1, 1]=1/(2*variance*variance)+exp.Rate*Math.Exp(mean+variance/2)/4;
            result.GetMeanAndVariance(out mean, out variance);
            if (double.IsInfinity(variance))
                            variance=1000;
            Vector theta = new Vector(new double[] { mean, variance });
            theta -= GammaFromShapeAndRate.twoByTwoInverse(hessianKL)*gradKL;
            result.SetMeanAndVariance(theta[0], theta[1]);
            return result; 
            ----------------------------------------------------------------- */
        }
    }
}
