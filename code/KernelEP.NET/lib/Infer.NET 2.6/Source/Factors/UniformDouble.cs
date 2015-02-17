using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MicrosoftResearch.Infer.Factors
{
    /// <summary>Provides outgoing messages for <see cref="Factor.UniformPlusMinus(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "UniformPlusMinus", typeof(double))]
    [Quality(QualityBand.Experimental)]
    public static class UniformPlusMinusOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <param name="upperBound">Incoming message from <c>upperBound</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(upperBound) p(upperBound) factor(sample,upperBound))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double sample, Pareto upperBound)
        {
            // factor is 1/(2*upperBound)
            // result is int_y^inf 1/(2*x)*s*L^s/x^(s+1) dx = 0.5*s*L^s/y^(s+1)/(s+1)
            // where y = max(sample,L)
            double x = Math.Abs(sample);
            double result = -MMath.Ln2 + Math.Log(upperBound.Shape / (upperBound.Shape + 1));
            if (upperBound.LowerBound < x)
            {
                result += upperBound.Shape * Math.Log(upperBound.LowerBound) - (upperBound.Shape + 1) * Math.Log(x);
            }
            else
            {
                result -= Math.Log(upperBound.LowerBound);
            }
            return result;
        }

        /// <summary>EP message to <c>upperBound</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <returns>The outgoing EP message to the <c>upperBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>upperBound</c> conditioned on the given values.</para>
        /// </remarks>
        public static Pareto UpperBoundAverageConditional(double sample)
        {
            return new Pareto(0, Math.Abs(sample));
        }

        /// <summary>VMP message to <c>upperBound</c>.</summary>
        /// <param name="sample">Constant value for <c>sample</c>.</param>
        /// <returns>The outgoing VMP message to the <c>upperBound</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>upperBound</c> conditioned on the given values.</para>
        /// </remarks>
        public static Pareto UpperBoundAverageLogarithm(double sample)
        {
            return UpperBoundAverageConditional(sample);
        }
    }
}
