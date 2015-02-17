// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Math.Log(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Log", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class LogOp_EP
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(log,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double log, double d)
        {
            return (log == Math.Log(d)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(log,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double log, double d)
        {
            return LogAverageFactor(log, d);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(log,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(double log, double d)
        {
            return LogAverageFactor(log, d);
        }

        //-- EP -------------------------------------------------------------------------------------------

        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(d) p(d) factor(log,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(double log, Gamma d)
        {
            double exp = Math.Exp(log);
            return d.GetLogProb(exp) + log;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Incoming message from <c>log</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(log) p(log) factor(log,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian log, double d)
        {
            return log.GetLogProb(Math.Log(d));
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Incoming message from <c>log</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_log">Outgoing message to <c>log</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(log,d) p(log,d) factor(log,d))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Gaussian log, Gamma d, [Fresh] Gaussian to_log)
        {
            Gamma g = Gamma.FromShapeAndRate(d.Shape + 1, d.Rate);
            return d.Shape / d.Rate * ExpOp.LogAverageFactor(g, log, to_log);
        }


        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Incoming message from <c>log</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_log">Outgoing message to <c>log</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(log,d) p(log,d) factor(log,d) / sum_log p(log) messageTo(log))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Gaussian log, Gamma d, [Fresh] Gaussian to_log)
        {
            return LogAverageFactor(log, d, to_log) - to_log.GetAverageLog(log);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(d) p(d) factor(log,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(double log, Gamma d)
        {
            return LogAverageFactor(log, d);
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="log">Incoming message from <c>log</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="to_log">Previous outgoing message to <c>log</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>d</c> as the random arguments are varied. The formula is <c>proj[p(d) sum_(log) p(log) factor(log,d)]/p(d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="log" /> is not a proper distribution.</exception>
        public static Gamma DAverageConditional([Proper] Gaussian log, Gamma d, Gaussian to_log)
        {
            var g = Gamma.FromShapeAndRate(d.Shape + 1, d.Rate);
            return ExpOp.ExpAverageConditional(g, log, to_log);
        }

        /// <summary>EP message to <c>d</c>.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <returns>The outgoing EP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma DAverageConditional(double log)
        {
            return Gamma.PointMass(Math.Exp(log));
        }

        /// <summary>EP message to <c>log</c>.</summary>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>The outgoing EP message to the <c>log</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>log</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian LogAverageConditional(double d)
        {
            return Gaussian.PointMass(Math.Log(d));
        }

        /// <summary>EP message to <c>log</c>.</summary>
        /// <param name="log">Incoming message from <c>log</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>log</c> as the random arguments are varied. The formula is <c>proj[p(log) sum_(d) p(d) factor(log,d)]/p(log)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="log" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian LogAverageConditional([Proper] Gaussian log, [SkipIfUniform] Gamma d, Gaussian result)
        {
            var g = Gamma.FromShapeAndRate(d.Shape + 1, d.Rate);
            return ExpOp.DAverageConditional(g, log, result);
        }

        /// <summary />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Gaussian LogAverageConditionalInit()
        {
            return Gaussian.Uniform();
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Math.Log(double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Math), "Log", typeof(double))]
    [Quality(QualityBand.Preview)]
    public class LogOp_VMP
    {
        //-- VMP -------------------------------------------------------------------------------------------

        /// <summary>
        /// Determines the amount of damping to use on the VMP updates for D. 
        /// </summary>
        public static double damping = 0.3;

        /// <summary>VMP message to <c>log</c>.</summary>
        /// <param name="log">Incoming message from <c>log</c>.</param>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>log</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gaussian LogAverageLogarithm(Gaussian log, double d, Gaussian result)
        {
            result.Point = Math.Log(d);
            return result;
        }

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma DAverageLogarithm(double log, Gamma d, Gamma result)
        {
            result.Point = Math.Exp(log);
            return result;
        }

        /// <summary>VMP message to <c>log</c>.</summary>
        /// <param name="log">Incoming message from <c>log</c>.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>log</c> as the random arguments are varied. The formula is <c>proj[sum_(d) p(d) factor(log,d)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gaussian LogAverageLogarithm(Gaussian log, [SkipIfUniform] Gamma d, Gaussian result)
        {
            if (d.IsPointMass)
                return LogAverageLogarithm(log, d.Point, result);
            result.SetMeanAndVariance(d.GetMeanLog(), MMath.Trigamma(d.Shape));
            return result;
        }

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="log">Incoming message from <c>log</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="d">Incoming message from <c>d</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_D">Previous outgoing message to <c>D</c>.</param>
        /// <returns>The outgoing VMP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> with <c>log</c> integrated out. The formula is <c>sum_log p(log) factor(log,d)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="log" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="d" /> is not a proper distribution.</exception>
        public static Gamma DAverageLogarithm([SkipIfUniform] Gaussian log, [SkipIfUniform] Gamma d, Gamma to_D)
        {
            if (log.IsPointMass)
                return DAverageLogarithm(log.Point, d, to_D);
            Vector grad = Vector.Zero(2);
            double meanLog = d.GetMeanLog();
            double m, v;
            log.GetMeanAndVariance(out m, out v);
            grad[0] = -MMath.Tetragamma(d.Shape) / (2 * v) - MMath.Trigamma(d.Shape) / v * (meanLog - m);
            grad[1] = (meanLog - m) / (v * d.Rate);
            Gamma approximateFactor = GammaFromShapeAndRateOp.NonconjugateProjection(d, grad);
            if (damping == 0.0)
                return approximateFactor;
            else
                return (approximateFactor ^ (1 - damping)) * (to_D ^ damping);
        }

#if false
		/// <summary>
		/// Evidence message for VMP
		/// </summary>
		/// <param name="log">Incoming message from 'log'.</param>
		/// <param name="d">Incoming message from 'd'.</param>
		/// <returns>Zero</returns>
		/// <remarks><para>
		/// In Variational Message Passing, the evidence contribution of a deterministic factor is zero.
		/// Adding up these values across all factors and variables gives the log-evidence estimate for VMP.
		/// </para></remarks>
		public static double AverageLogFactor(Gaussian log, Gamma d)
		{
			double m, v;
			log.GetMeanAndVariance(out m, out v);
			double Elogd=d.GetMeanLog();
			double Elogd2;
			if (!d.IsPointMass)
				Elogd2 = MMath.Trigamma(d.Shape) + Elogd * Elogd;
			else
				Elogd2 = Math.Log(d.Point) * Math.Log(d.Point);
			return -Elogd2/(2*v)+m*Elogd/v-m*m/(2*v)-MMath.LnSqrt2PI-.5*Math.Log(v);
		}
#endif

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(log,d))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor()
        {
            return 0;
        }

        /// <summary>VMP message to <c>log</c>.</summary>
        /// <param name="d">Constant value for <c>d</c>.</param>
        /// <returns>The outgoing VMP message to the <c>log</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>log</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma LogAverageLogarithm(double d)
        {
            return Gamma.PointMass(Math.Log(d));
        }

        private const string NotSupportedMessage = "VMP cannot support deterministic factors such as Log with fixed outputs";

        /// <summary>VMP message to <c>d</c>.</summary>
        /// <param name="log">Constant value for <c>log</c>.</param>
        /// <returns>The outgoing VMP message to the <c>d</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>d</c> conditioned on the given values.</para>
        /// </remarks>
        [NotSupported(NotSupportedMessage)]
        public static Gaussian DAverageLogarithm(double log)
        {
            throw new NotSupportedException(NotSupportedMessage);
        }
    }
}
