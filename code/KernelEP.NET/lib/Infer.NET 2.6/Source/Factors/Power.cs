using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions;

namespace MicrosoftResearch.Infer.Factors
{
    /// <summary>Provides outgoing messages for <see cref="Math.Pow(double, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Pow", typeof(double), typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class PowerOp
    {
        /// <summary>EP message to <c>pow</c>.</summary>
        /// <param name="x">Incoming message from <c>x</c>.</param>
        /// <param name="y">Constant value for <c>y</c>.</param>
        /// <returns>The outgoing EP message to the <c>pow</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>pow</c> as the random arguments are varied. The formula is <c>proj[p(pow) sum_(x) p(x) factor(pow,x,y)]/p(pow)</c>.</para>
        /// </remarks>
        public static Pareto PowAverageConditional(Pareto x, double y)
        {
            if (y < 0)
                throw new NotSupportedException("Pareto raised to a negative power (" + y + ") cannot be represented by a Pareto distribution");
            // p(x) =propto 1/x^(s+1)
            // z = x^y implies
            // p(z) = p(x = z^(1/y)) (1/y) z^(1/y-1)
            //      =propto z^(-(s+1)/y+1/y-1) = z^(-s/y-1)
            return new Pareto(x.Shape / y, Math.Pow(x.LowerBound, y));
        }

        /// <summary>EP message to <c>x</c>.</summary>
        /// <param name="pow">Incoming message from <c>pow</c>.</param>
        /// <param name="x">Incoming message from <c>x</c>.</param>
        /// <param name="y">Constant value for <c>y</c>.</param>
        /// <returns>The outgoing EP message to the <c>x</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>x</c> as the random arguments are varied. The formula is <c>proj[p(x) sum_(pow) p(pow) factor(pow,x,y)]/p(x)</c>.</para>
        /// </remarks>
        public static Gamma XAverageConditional(Pareto pow, Gamma x, double y)
        {
            // factor is delta(pow - x^y)
            // marginal for x is Pareto(x^y; s,L) Ga(x; a,b)
            // =propto x^(a-1-y(s+1)) exp(-bx)   for x >= L^(1/y)
            // we can compute moments via the incomplete Gamma function.
            // change variables to: z=bx,  dz=b dx
            // int_LL^inf x^(c-1) exp(-bx) dx = int_(bLL)^inf z^(c-1)/b^c exp(-z) dz = gammainc(c,bLL,inf)/b^c
            double lowerBound = Math.Pow(pow.LowerBound, 1 / y);
            double b = x.Rate;
            double c = x.Shape - y * (pow.Shape + 1);
            double bL = b * lowerBound;
            double m, m2;
            if (y > 0)
            {
                // note these ratios can be simplified
                double z = MMath.GammaUpper(c, bL);
                m = MMath.GammaUpper(c + 1, bL) * c / z / b;
                m2 = MMath.GammaUpper(c + 2, bL) * c * (c + 1) / z / (b * b);
            }
            else
            {
                double z = MMath.GammaLower(c, bL);
                m = MMath.GammaLower(c + 1, bL) * c / z / b;
                m2 = MMath.GammaLower(c + 2, bL) * c * (c + 1) / z / (b * b);
            }
            double v = m2 - m * m;
            Gamma xPost = Gamma.FromMeanAndVariance(m, v);
            Gamma result = new Gamma();
            result.SetToRatio(xPost, x, true);
            return result;
        }
    }
}
