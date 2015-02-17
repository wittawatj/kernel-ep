using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Distributions;

namespace MicrosoftResearch.Infer.Factors
{
    /// <summary>Provides outgoing messages for <see cref="Factor.Double(int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Double", typeof(int))]
    [Quality(QualityBand.Preview)]
    public static class DoubleOp
    {
        public static bool ForceProper = true;

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Double">Constant value for <c>double</c>.</param>
        /// <param name="Integer">Constant value for <c>integer</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(double,integer))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double Double, int Integer)
        {
            return (Double == Factor.Double(Integer)) ? 0.0 : double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Double">Constant value for <c>double</c>.</param>
        /// <param name="Integer">Incoming message from <c>integer</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(integer) p(integer) factor(double,integer))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double Double, Discrete Integer)
        {
            int i = (int)Double;
            if (i != Double || i < 0 || i >= Integer.Dimension)
                return double.NegativeInfinity;
            return Integer.GetLogProb(i);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Double">Incoming message from <c>double</c>.</param>
        /// <param name="Integer">Constant value for <c>integer</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(double) p(double) factor(double,integer) / sum_double p(double) messageTo(double))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Gaussian Double, int Integer)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Double">Incoming message from <c>double</c>.</param>
        /// <param name="Integer">Incoming message from <c>integer</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(double,integer) p(double,integer) factor(double,integer))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian Double, Discrete Integer)
        {
            double logZ = double.NegativeInfinity;
            for (int i = 0; i < Integer.Dimension; i++)
            {
                double logp = Double.GetLogProb(i) + Integer.GetLogProb(i);
                logZ = MMath.LogSumExp(logZ, logp);
            }
            return logZ;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Double">Incoming message from <c>double</c>.</param>
        /// <param name="Integer">Incoming message from <c>integer</c>.</param>
        /// <param name="to_double">Previous outgoing message to <c>double</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(double,integer) p(double,integer) factor(double,integer) / sum_double p(double) messageTo(double))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Gaussian Double, Discrete Integer, Gaussian to_double)
        {
            return LogAverageFactor(Double, Integer) - to_double.GetLogAverageOf(Double);
        }

        /// <summary>EP message to <c>double</c>.</summary>
        /// <param name="Double">Incoming message from <c>double</c>.</param>
        /// <param name="Integer">Incoming message from <c>integer</c>.</param>
        /// <returns>The outgoing EP message to the <c>double</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>double</c> as the random arguments are varied. The formula is <c>proj[p(double) sum_(integer) p(integer) factor(double,integer)]/p(double)</c>.</para>
        /// </remarks>
        public static Gaussian DoubleAverageConditional(Gaussian Double, Discrete Integer)
        {
            if (Integer.IsPointMass)
                return Gaussian.PointMass(Factor.Double(Integer.Point));
            // Z = sum_i int_x q(x) delta(x - i) q(i) dx
            //   = sum_i q(x=i) q(i)
            double max = double.NegativeInfinity;
            for (int i = 0; i < Integer.Dimension; i++)
            {
                double logp = Double.GetLogProb(i);
                if (logp > max)
                    max = logp;
            }
            if (double.IsNegativeInfinity(max))
                throw new AllZeroException();
            GaussianEstimator est = new GaussianEstimator();
            for (int i = 0; i < Integer.Dimension; i++)
            {
                double logp = Double.GetLogProb(i);
                est.Add(i, Integer[i]*Math.Exp(logp - max));
            }
            Gaussian result = est.GetDistribution(new Gaussian());
            result.SetToRatio(result, Double, ForceProper);
            return result;
        }

        /// <summary>EP message to <c>integer</c>.</summary>
        /// <param name="Double">Incoming message from <c>double</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>integer</c> as the random arguments are varied. The formula is <c>proj[p(integer) sum_(double) p(double) factor(double,integer)]/p(integer)</c>.</para>
        /// </remarks>
        public static Discrete IntegerAverageConditional(Gaussian Double, Discrete result)
        {
            if (Double.IsPointMass)
            {
                result.Point = (int)Math.Round(Double.Point);
                return result;
            }
            Vector probs = result.GetWorkspace();
            double max = double.NegativeInfinity;
            for (int i = 0; i < result.Dimension; i++)
            {
                double logp = Double.GetLogProb(i);
                probs[i] = logp;
                if (logp > max)
                    max = logp;
            }
            if (double.IsNegativeInfinity(max))
                throw new AllZeroException();
            probs.SetToFunction(probs, logp => Math.Exp(logp - max));
            result.SetProbs(probs);
            return result;
        }
    }
}
