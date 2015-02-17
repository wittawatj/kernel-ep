// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using MicrosoftResearch.Infer.Distributions;

    internal static class BufferTester
    {
        [Hidden]
        public static T Copy<T>(T value)
        {
            return value;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="BufferTester.Copy{T}(T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(BufferTester), "Copy<>")]
    [Buffers("buffer")]
    [Quality(QualityBand.Experimental)]
    public static class BufferTesterCopyOp
    {
        /// <summary>Update the buffer <c>buffer</c>.</summary>
        /// <param name="copy">Incoming message from <c>copy</c>.</param>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T Buffer<T>(T copy, T value, T result)
        {
            return value;
        }

        /// <summary>Initialize the buffer <c>buffer</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <returns>Initial value of buffer <c>buffer</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T BufferInit<T>(T value)
        {
            return value;
        }

        /// <summary>EP message to <c>copy</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>.</param>
        /// <param name="buffer">Buffer <c>buffer</c>.</param>
        /// <returns>The outgoing EP message to the <c>copy</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>copy</c> as the random arguments are varied. The formula is <c>proj[p(copy) sum_(value) p(value) factor(copy,value)]/p(copy)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T CopyAverageConditional<T>(T value, T buffer)
        {
            return value;
        }

        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="copy">Incoming message from <c>copy</c>.</param>
        /// <returns>The outgoing EP message to the <c>value</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>value</c> as the random arguments are varied. The formula is <c>proj[p(value) sum_(copy) p(copy) factor(copy,value)]/p(value)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageConditional<T>(T copy)
        {
            return copy;
        }
    }

    /// <summary>
    /// Buffer factors
    /// </summary>
    internal static class Buffer
    {
        /// <summary>
        /// Value factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        [Hidden]
        public static T Value<T>()
        {
            return default(T);
        }

        /// <summary>
        /// Infer factor
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        [Hidden]
        public static void Infer<T>(T value)
        {
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Buffer.Value{T}()" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Buffer), "Value<>")]
    [Quality(QualityBand.Mature)]
    internal static class BufferOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing EP message to the <c>value</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageConditional<T>([SkipIfUniform] T value)
        {
            return value;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>value</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageLogarithm<T>([SkipIfUniform] T value)
        {
            return value;
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Buffer.Infer{T}(T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Buffer), "Infer<>")]
    [Quality(QualityBand.Experimental)]
    internal static class InferOp
    {
        /// <summary>EP message to <c>value</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageConditional<T>([SkipIfUniform] T value, T result)
            where T : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        /// <summary>VMP message to <c>value</c>.</summary>
        /// <param name="value">Incoming message from <c>value</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>value</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="value" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the value.</typeparam>
        public static T ValueAverageLogarithm<T>([SkipIfUniform] T value, T result)
            where T : SettableToUniform
        {
            return ValueAverageConditional(value, result);
        }
    }
}
