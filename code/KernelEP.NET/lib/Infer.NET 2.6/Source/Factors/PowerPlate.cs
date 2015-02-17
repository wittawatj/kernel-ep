// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// Power plate factor method
    /// </summary>
    [Hidden]
    public static class PowerPlate
    {
        /// <summary>
        /// Copy a value from outside to the inside of a power plate.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="exponent"></param>
        /// <returns>A copy of value.</returns>
        public static T Enter<T>([IsReturned] T value, double exponent)
        {
            return value;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="PowerPlate.Enter{T}(T, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(PowerPlate), "Enter<>")]
    [Quality(QualityBand.Preview)]
    public static class PowerPlateOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="exponent">Constant value for <c>exponent</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(enter) p(enter) factor(enter,value,exponent)]/p(value)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enter" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static T ValueAverageConditional<T>([SkipIfUniform] T enter, double exponent, T result)
            where T : SettableToPower<T>
        {
            result.SetToPower(enter, exponent);
            return result;
        }

        /// <summary>EP message to <c>enter</c>.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="exponent">Constant value for <c>exponent</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enter</c> as the random arguments are varied. The formula is <c>proj[p(enter) sum_(value) p(value) factor(enter,value,exponent)]/p(enter)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        [SkipIfAllUniform]
        public static T EnterAverageConditional<T>([NoInit, Cancels] T enter, T value, double exponent, T result)
            where T : SettableToPower<T>, SettableToProduct<T>
        {
            if (exponent == 0)
            {
                // it doesn't matter what we return in this case, so we return something proper
                // to avoid spurious improper message exceptions
                result.SetToPower(value, 1.0);
            }
            else
            {
                // to_enter = value*enter^(exponent-1)
                result.SetToPower(enter, exponent - 1);
                result.SetToProduct(value, result);
            }
            return result;
        }

        /// <summary />
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        [Skip]
        public static T EnterAverageConditionalInit<T>([IgnoreDependency] T value)
            where T : ICloneable
        {
            return (T)value.Clone();
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="exponent">Constant value for <c>exponent</c>.</param>
        /// <param name="to_enter">Outgoing message to <c>enter</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(enter,value) p(enter,value) factor(enter,value,exponent) / sum_enter p(enter) messageTo(enter))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static double LogEvidenceRatio<T>(T enter, T value, double exponent, [Fresh] T to_enter)
            where T : CanGetLogAverageOf<T>, CanGetLogAverageOfPower<T>
        {
            // qnot(x) =propto q(x)/m_out(x)
            // qnot2(x) =propto q(x)/m_out(x)^n
            // the interior of the plate sends (int_x qnot(x) f(x) dx)^n
            // which is missing (int_x qnot2(x) m_out(x)^n dx)/(int_x qnot(x) m_out(x) dx)^n
            // this factor sends the missing piece, where:
            // enter = m_out(x)
            // to_enter = qnot(x)
            // value = qnot2(x)
            return value.GetLogAverageOfPower(enter, exponent) - exponent * to_enter.GetLogAverageOf(enter);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(enter,value,exponent))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="enter">Incoming message from <c>enter</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="exponent">Constant value for <c>exponent</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> with <c>enter</c> integrated out. The formula is <c>sum_enter p(enter) factor(enter,value,exponent)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="enter" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static T ValueAverageLogarithm<T>([SkipIfUniform] T enter, double exponent, T result)
            where T : SettableToPower<T>
        {
            return ValueAverageConditional<T>(enter, exponent, result);
        }

        /// <summary>VMP message to <c>enter</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <returns>The outgoing VMP message to the <c>enter</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>enter</c> as the random arguments are varied. The formula is <c>proj[sum_(value) p(value) factor(enter,value,exponent)]</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the distribution over the variable entering the power plate.</typeparam>
        public static T EnterAverageLogarithm<T>([IsReturned] T value)
        {
            return value;
        }
    }

    /// <summary>
    /// Damp factor methods
    /// </summary>
    public static class Damp
    {
        /// <summary>
        /// Copy a value and damp the backward message.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="stepsize">1.0 means no damping, 0.0 is infinite damping.</param>
        /// <returns></returns>
        /// <remarks>
        /// If you use this factor, be sure to increase the number of algorithm iterations appropriately.
        /// The number of iterations should increase according to the reciprocal of stepsize.
        /// </remarks>
        public static T Backward<T>([IsReturned] T value, double stepsize)
        {
            return value;
        }

        /// <summary>
        /// Copy a value and damp the forward message.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="stepsize">1.0 means no damping, 0.0 is infinite damping.</param>
        /// <returns></returns>
        /// <remarks>
        /// If you use this factor, be sure to increase the number of algorithm iterations appropriately.
        /// The number of iterations should increase according to the reciprocal of stepsize.
        /// </remarks>
        public static T Forward<T>([IsReturned] T value, double stepsize)
        {
            return value;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Damp.Backward{T}(T, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Damp), "Backward<>")]
    [Quality(QualityBand.Preview)]
    public static class DampBackwardOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(backward,value,stepsize))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

           /// <summary>EP message to <c>value</c>.</summary>
           /// <param name="backward">Incoming message from <c>backward</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
           /// <param name="stepsize">Constant value for <c>stepsize</c>.</param>
           /// <param name="to_value">Previous outgoing message to <c>value</c>.</param>
           /// <returns>The outgoing EP message to the <c>value</c> argument.</returns>
           /// <remarks>
           ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(backward) p(backward) factor(backward,value,stepsize)]/p(value)</c>.</para>
           /// </remarks>
           /// <exception cref="ImproperMessageException">
           ///   <paramref name="backward" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ValueAverageConditional<Distribution>(
            [SkipIfUniform] Distribution backward, double stepsize, Distribution to_value)
            where Distribution : SettableToPower<Distribution>, SettableToProduct<Distribution>
        {
            // damp the backward message.
            // to_value holds the last message to value.
            // result = to_value^(1-stepsize) * backward^stepsize
            Distribution result = to_value;
            result.SetToPower(to_value, (1 - stepsize) / stepsize);
            result.SetToProduct(result, backward);
            result.SetToPower(result, stepsize);
            return result;
        }

        /// <summary>EP message to <c>backward</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <returns>The outgoing EP message to the <c>backward</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>backward</c> as the random arguments are varied. The formula is <c>proj[p(backward) sum_(value) p(value) factor(backward,value,stepsize)]/p(backward)</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution BackwardAverageConditional<Distribution>([IsReturned] Distribution value)
        {
            return value;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Damp.Forward{T}(T, double)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Damp), "Forward<>")]
    [Quality(QualityBand.Preview)]
    public static class DampForwardOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(forward,value,stepsize))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio()
        {
            return 0.0;
        }

           /// <summary>EP message to <c>forward</c>.</summary>
           /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
           /// <param name="stepsize">Constant value for <c>stepsize</c>.</param>
           /// <param name="to_forward">Previous outgoing message to <c>forward</c>.</param>
           /// <returns>The outgoing EP message to the <c>forward</c> argument.</returns>
           /// <remarks>
           ///   <para>The outgoing message is a distribution matching the moments of <c>forward</c> as the random arguments are varied. The formula is <c>proj[p(forward) sum_(value) p(value) factor(forward,value,stepsize)]/p(forward)</c>.</para>
           /// </remarks>
           /// <exception cref="ImproperMessageException">
           ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ForwardAverageConditional<Distribution>(
            [SkipIfUniform] Distribution value, double stepsize, Distribution to_forward)
            where Distribution : SettableToPower<Distribution>, SettableToProduct<Distribution>
        {
            // damp the backward message.
            // to_forward holds the last message to value.
            // result = to_forward^(1-stepsize) * backward^stepsize
            Distribution result = to_forward;
            result.SetToPower(to_forward, (1 - stepsize) / stepsize);
            result.SetToProduct(result, value);
            result.SetToPower(result, stepsize);
            return result;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="forward">Incoming message from <c>forward</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(forward) p(forward) factor(forward,value,stepsize)]/p(value)</c>.</para>
        /// </remarks>
        /// <typeparam name="Distribution">The type of the distribution over the damped variable.</typeparam>
        public static Distribution ValueAverageConditional<Distribution>([IsReturned] Distribution forward, Distribution result)
            where Distribution : SettableTo<Distribution>
        {
            result.SetTo(forward);
            return result;
        }
    }
}
