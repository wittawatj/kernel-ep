// (C) Copyright 2008 Microsoft Research Cambridge

#define SpecializeArrays
#define MinimalGenericTypeParameters

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Replicate{T}(T, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Replicate<>", Default = true)]
    [Buffers("marginal", "toDef")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateOp_Divide
    {
        /// <summary>EP message to <c>Def</c>.</summary>
        /// <param name="toDef">Buffer <c>toDef</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Def</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T DefAverageConditional<T>([IsReturned] T toDef, T result)
            where T : SettableTo<T>
        {
            result.SetTo(toDef);
            return result;
        }

        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="marginal">Buffer <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Uses</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // Uses is marked Cancels because the forward message does not really depend on the backward message
        public static T UsesAverageConditional<T>([Indexed, Cancels] T Uses, [SkipIfUniform] T marginal, int resultIndex, T result)
            where T : SettableToRatio<T>
        {
            result.SetToRatio(marginal, Uses);
            return result;
        }

        /// <summary>Initialize the buffer <c>marginal</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Initial value of buffer <c>marginal</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalInit<T>([SkipIfUniform] T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <summary>Update the buffer <c>marginal</c>.</summary>
        /// <param name="toDef">Buffer <c>toDef</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        [Fresh]
        public static T Marginal<T>([NoInit] T toDef, T Def, T result)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(Def, toDef);
            return result;
        }

        /// <summary />
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="use" />
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // SkipIfUniform on 'use' causes this line to be pruned when the backward message isn't changing
        [SkipIfAllUniform]
        [MultiplyAll]
        [Fresh]
        public static T MarginalIncrement<T>(T result, [InducedSource] T def, [SkipIfUniform, InducedTarget] T use)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(use, def);
            return result;
        }

        /// <summary>Initialize the buffer <c>toDef</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns>Initial value of buffer <c>toDef</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip] // this is needed to instruct the scheduler to treat the buffer as uninitialized
        public static T ToDefInit<T>(T Def)
            where T : ICloneable, SettableToUniform
        {
            // must construct from Def instead of Uses because Uses array may be empty
            return ArrayHelper.MakeUniform(Def);
        }

        /// <summary>Update the buffer <c>toDef</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        [Fresh]
        // NoInit is required for PlusProductHierarchyTest
        public static T ToDef<T>([NoInit] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Replicate{T}(T, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Replicate<>", Default = false)]
    [Buffers("marginal")]
    [Quality(QualityBand.Preview)]
    public static class Replicate2BufferOp
    {
#if false
    /// <summary>
    /// EP message to 'Uses'
    /// </summary>
    /// <param name="Uses">Incoming message from 'Uses'.</param>
    /// <param name="marginal">Buffer 'marginal'.</param>
    /// <param name="result">Modified to contain the outgoing message</param>
    /// <returns><paramref name="result"/></returns>
    /// <remarks><para>
    /// The outgoing message is the factor viewed as a function of 'Uses' conditioned on the given values.
    /// </para></remarks>
    //[SkipIfAllUniform]
		public static TListRet UsesAverageConditional<T, TList, TListRet>(TList Uses, [Fresh, SkipIfUniform] T marginal, TListRet result)
			where TList : IList<T>
			where TListRet : IList<T>
			where T : SettableToRatio<T>
		{
			for (int i = 0; i < result.Count; i++) {
				T dist = result[i];
				dist.SetToRatio(marginal, Uses[i]);
				result[i] = dist;
			}
			return result;
		}
#endif

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="marginal">Buffer <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,Count)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>(
            [MatchingIndex, IgnoreDependency] IList<T> Uses, // Uses dependency must be ignored for Sequential schedule
            [IgnoreDependency, SkipIfUniform] T Def,
            [Fresh, SkipIfUniform] T marginal,
            int resultIndex,
            T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Count == 1)
            {
                result.SetTo(Def);
                return result;
            }
            if (true)
            {
                try
                {
                    result.SetToRatio(marginal, Uses[resultIndex]);
                    return result;
                }
                catch (DivideByZeroException)
                {
                    return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
                }
            }
            else
            {
                // check that ratio is same as product
                result.SetToRatio(marginal, Uses[resultIndex]);
                T result2 = (T)((ICloneable)result).Clone();
                ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result2);
                double err = ((Diffable)result).MaxDiff(result2);
                if (err > 1e-4)
                    Console.WriteLine(err);
                return result;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

#if SpecializeArrays
        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="marginal">Buffer <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,Count)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>(
            [MatchingIndex, IgnoreDependency] T[] Uses, // Uses dependency must be ignored for Sequential schedule
            [IgnoreDependency, SkipIfUniform] T Def,
            [Fresh, SkipIfUniform] T marginal,
            int resultIndex,
            T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Length == 1)
            {
                result.SetTo(Def);
                return result;
            }
            try
            {
                result.SetToRatio(marginal, Uses[resultIndex]);
                return result;
            }
            catch (DivideByZeroException)
            {
                return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
            }
        }
#endif
        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="count">Constant value for <c>Count</c>.</param>
        /// <param name="factory" />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        /// <typeparam name="ArrayType">The type of arrays produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static ArrayType UsesAverageConditionalInit<T, ArrayType>(
            [IgnoreDependency] T Def, int count, IArrayFactory<T, ArrayType> factory)
            where T : ICloneable
        {
            return factory.CreateArray(count, i => (T)Def.Clone());
        }

        /// <summary>Initialize the buffer <c>marginal</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Initial value of buffer <c>marginal</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [Skip] // this is needed to instruct the scheduler to treat marginal as uninitialized
        public static T MarginalInit<T>([SkipIfUniform] T Def)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <summary>Update the buffer <c>marginal</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T Marginal<T>(IList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            return ReplicateOp_NoDivide.MarginalAverageConditional(Uses, Def, result);
        }

        /// <summary />
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <param name="use" />
        /// <param name="def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        public static T MarginalIncrement<T>(T result, [SkipIfUniform] T use, [SkipIfUniform] T def)
            where T : SettableToProduct<T>
        {
            result.SetToProduct(use, def);
            return result;
        }

#if MinimalGenericTypeParameters
        /// <summary>EP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Def</c> as the random arguments are varied. The formula is <c>proj[p(Def) sum_(Uses) p(Uses) factor(Uses,Def,Count)]/p(Def)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }

#if SpecializeArrays
        /// <summary>EP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Def</c> as the random arguments are varied. The formula is <c>proj[p(Def) sum_(Uses) p(Uses) factor(Uses,Def,Count)]/p(Def)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#else
			public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] IList<TUses> Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#if SpecializeArrays
        public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] TUses[] Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#endif
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.ReplicateWithMarginal{T}(T, int, out T)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "ReplicateWithMarginal<>", Default = true)]
    [Quality(QualityBand.Preview)]
    public static class ReplicateBufferOp
    {
        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_marginal">Outgoing message to <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,count,Marginal)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] IList<T> Uses, [SkipIfUniform] T Def, [SkipIfUniform, Fresh] T to_marginal, int resultIndex, T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Count == 1)
            {
                result.SetTo(Def);
                return result;
            }
            try
            {
                result.SetToRatio(to_marginal, Uses[resultIndex]);
                return result;
            }
            catch (DivideByZeroException)
            {
                return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
            }
        }

#if SpecializeArrays
        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_marginal">Outgoing message to <c>marginal</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,count,Marginal)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        //[SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] T[] Uses, [SkipIfUniform] T Def, [SkipIfUniform, Fresh] T to_marginal, int resultIndex, T result)
            where T : SettableToRatio<T>, SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Length == 1)
            {
                result.SetTo(Def);
                return result;
            }
            try
            {
                result.SetToRatio(to_marginal, Uses[resultIndex]);
                return result;
            }
            catch (DivideByZeroException)
            {
                return ReplicateOp_NoDivide.UsesAverageConditional(Uses, Def, resultIndex, result);
            }
        }
#endif
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Replicate{T}(T, int)" /></description></item><item><description><see cref="Factor.ReplicateWithMarginal{T}(T, int, out T)" /></description></item></list>, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the variable being replicated.</typeparam>
    [FactorMethod(typeof(Factor), "Replicate<>", Default = true)]
    [FactorMethod(typeof(Factor), "ReplicateWithMarginal<>", Default = true)]
    [Quality(QualityBand.Mature)]
    public static class ReplicateGibbsOp<T>
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,Count) / sum_Uses p(Uses) messageTo(Uses))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static double LogEvidenceRatio<TDist>([SkipIfAllUniform] IList<TDist> Uses, T Def)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, T Def)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

#if false
		public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, int resultIndex, T result)
			where TDist : IDistribution<T>, Sampleable<T>
		{
			return marginal.LastSample;
		}
#elif false
		public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, TDist def, int resultIndex, T result)
			where TDist : IDistribution<T>, Sampleable<T>
		{
			return marginal.LastSample;
		}
		public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> marginal, T def, int resultIndex, T result)
			where TDist : IDistribution<T>, Sampleable<T>
		{
			if (def is bool[]) {
				if (!Util.ValueEquals((bool[])(object)def, (bool[])(object)marginal.LastSample)) throw new Exception("gotcha");
			}
			else if (def is double[]) {
				if (!Util.ValueEquals((double[])(object)def, (double[])(object)marginal.LastSample)) throw new Exception("gotcha");
			} else if (!def.Equals(marginal.LastSample)) throw new Exception("gotcha");
			return marginal.LastSample;
		}
#else
        /// <summary />
        /// <param name="to_marginal" />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T UsesGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, TDist def, int resultIndex, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            // This method must depend on Def, even though Def isn't used, in order to get the right triggers
            return to_marginal.LastSample;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static T UsesGibbs([IsReturned] T def, int resultIndex, T result)
        {
            return def;
        }
#endif

#if true
        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        // until .NET 4
        public static TDist UsesGibbs<TDist>([IsReturned] TDist def, int resultIndex, TDist result)
            where TDist : IDistribution<T>
        {
            return def;
        }
#endif

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }

#if SpecializeArrays
        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] TDist[] Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableToProduct<TDist>, SettableTo<TDist>
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }
#endif

        /// <summary />
        /// <param name="to_marginal" />
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T DefGibbs<TDist>([SkipIfUniform] GibbsMarginal<TDist, T> to_marginal, T result)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return to_marginal.LastSample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="to_marginal" />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic]
        [SkipIfAllUniform]
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            IList<TDist> Uses,
            [SkipIfUniform] TDist Def,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, SettableToProduct<TDist>, SettableToRatio<TDist>, SettableTo<TDist>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.SetTo(Def);
            marginal = Distribution.SetToProductWithAll(marginal, Uses);
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="to_marginal" />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic] // must be labelled Stochastic to get correct schedule, even though it isn't Stochastic
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            T Def,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            marginal.Point = Def;
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="to_marginal" />
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic] // must be labelled Stochastic to get correct schedule, even though it isn't Stochastic
        public static GibbsMarginal<TDist, T> MarginalGibbs<TDist>(
            T[] Uses,
            GibbsMarginal<TDist, T> to_marginal)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            GibbsMarginal<TDist, T> result = to_marginal;
            TDist marginal = result.LastConditional;
            if (Uses.Length != 1)
                throw new ArgumentException("Uses.Length (" + Uses.Length + ") != 1");
            marginal.Point = Uses[0];
            result.LastConditional = marginal;
            // Allow a sample to be drawn from the last conditional, and add it to the sample
            // list and conditional list
            result.PostUpdate();
            return result;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static GibbsMarginal<TDist, T> MarginalGibbsInit<TDist>([IgnoreDependency] TDist def)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return new GibbsMarginal<TDist, T>(def, 100, 1, true, true, true);
        }
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.ReplicateWithMarginalGibbs{T}(T, int, int, int, out T, out T, out T)" />, given random arguments to the function.</summary>
    /// <typeparam name="T">The type of the replicated variable.</typeparam>
    [FactorMethod(typeof(Factor), "ReplicateWithMarginalGibbs<>")]
    [Buffers("sample", "conditional", "marginalEstimator", "sampleAcc", "conditionalAcc")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateGibbsOp2<T>
    {
        /// <summary>Initialize the buffer <c>conditional</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <returns>Initial value of buffer <c>conditional</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static TDist ConditionalInit<TDist>([IgnoreDependency] TDist to_marginal)
            where TDist : ICloneable
        {
            return (TDist)to_marginal.Clone();
        }

        /// <summary>Update the buffer <c>conditional</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist Conditional<TDist>(T Def, TDist result)
            where TDist : HasPoint<T>
        {
            result.Point = Def;
            return result;
        }

        /// <summary>Update the buffer <c>conditional</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist Conditional<TDist>(IList<TDist> Uses, [SkipIfAnyUniform] TDist Def, TDist result)
            where TDist : SettableTo<TDist>, SettableToProduct<TDist>
        {
            result.SetTo(Def);
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }

        /// <summary>Update the buffer <c>sample</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="conditional">Buffer <c>conditional</c>.</param>
        /// <returns>New value of buffer <c>sample</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Stochastic]
        public static T Sample<TDist>([IgnoreDependency] TDist to_marginal, [Proper] TDist conditional)
            where TDist : IDistribution<T>, Sampleable<T>
        {
            return conditional.Sample();
        }

        /// <summary>Initialize the buffer <c>marginalEstimator</c>.</summary>
        /// <param name="to_marginal">Previous outgoing message to <c>marginal</c>.</param>
        /// <param name="burnIn">Constant value for <c>burnIn</c>.</param>
        /// <returns>Initial value of buffer <c>marginalEstimator</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static BurnInAccumulator<TDist> MarginalEstimatorInit<TDist>([IgnoreDependency] TDist to_marginal, int burnIn)
            where TDist : IDistribution<T>
        {
            Accumulator<TDist> est = (Accumulator<TDist>)ArrayEstimator.CreateEstimator<TDist, T>(to_marginal, true);
            return new BurnInAccumulator<TDist>(burnIn, 1, est);
        }

        /// <summary>Update the buffer <c>marginalEstimator</c>.</summary>
        /// <param name="conditional">Buffer <c>conditional</c>.</param>
        /// <param name="marginalEstimator">Buffer <c>marginalEstimator</c>.</param>
        /// <returns>New value of buffer <c>marginalEstimator</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        /// <typeparam name="TAcc">The type of the marginal estimator.</typeparam>
        public static TAcc MarginalEstimator<TDist, TAcc>([Proper] TDist conditional, TAcc marginalEstimator)
            where TAcc : Accumulator<TDist>
        {
            marginalEstimator.Add(conditional);
            return marginalEstimator;
        }

        /// <summary />
        /// <param name="marginalEstimator">Buffer <c>marginalEstimator</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist MarginalGibbs<TDist>(BurnInAccumulator<TDist> marginalEstimator, TDist result)
        {
            return ((Estimator<TDist>)marginalEstimator.Accumulator).GetDistribution(result);
        }

        /// <summary>Initialize the buffer <c>sampleAcc</c>.</summary>
        /// <param name="to_samples">Previous outgoing message to <c>samples</c>.</param>
        /// <param name="burnIn">Constant value for <c>burnIn</c>.</param>
        /// <param name="thin">Constant value for <c>thin</c>.</param>
        /// <returns>Initial value of buffer <c>sampleAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Accumulator<T> SampleAccInit(ICollection<T> to_samples, int burnIn, int thin)
        {
            return new BurnInAccumulator<T>(burnIn, thin, new AccumulateIntoCollection<T>(to_samples));
        }

        /// <summary>Update the buffer <c>sampleAcc</c>.</summary>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <param name="sampleAcc">Buffer <c>sampleAcc</c>.</param>
        /// <returns>New value of buffer <c>sampleAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static Accumulator<T> SampleAcc(T sample, Accumulator<T> sampleAcc)
        {
            sampleAcc.Add(sample);
            return sampleAcc;
        }

        /// <summary />
        /// <param name="sampleAcc">Buffer <c>sampleAcc</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TList">The type of the outgoing message.</typeparam>
        public static TList SamplesGibbs<TList>(Accumulator<T> sampleAcc, TList result)
            where TList : ICollection<T>
        {
            // do nothing since result was already modified by sampleAcc
            return result;
        }

        /// <summary>Initialize the buffer <c>conditionalAcc</c>.</summary>
        /// <param name="to_conditionals">Previous outgoing message to <c>conditionals</c>.</param>
        /// <param name="burnIn">Constant value for <c>burnIn</c>.</param>
        /// <param name="thin">Constant value for <c>thin</c>.</param>
        /// <returns>Initial value of buffer <c>conditionalAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static Accumulator<TDist> ConditionalAccInit<TDist>(ICollection<TDist> to_conditionals, int burnIn, int thin)
        {
            return new BurnInAccumulator<TDist>(burnIn, thin, new AccumulateIntoCollection<TDist>(to_conditionals));
        }

        /// <summary>Update the buffer <c>conditionalAcc</c>.</summary>
        /// <param name="conditional">Buffer <c>conditional</c>.</param>
        /// <param name="conditionalAcc">Buffer <c>conditionalAcc</c>.</param>
        /// <returns>New value of buffer <c>conditionalAcc</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static Accumulator<TDist> ConditionalAcc<TDist>(TDist conditional, Accumulator<TDist> conditionalAcc)
            where TDist : ICloneable
        {
            conditionalAcc.Add((TDist)conditional.Clone());
            return conditionalAcc;
        }

        /// <summary />
        /// <param name="conditionalAcc">Buffer <c>conditionalAcc</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        /// <typeparam name="TDistList">The type of the outgoing message.</typeparam>
        public static TDistList ConditionalsGibbs<TDist, TDistList>(Accumulator<TDist> conditionalAcc, TDistList result)
            where TDistList : ICollection<TDist>
        {
            // do nothing since result was already modified by Acc
            return result;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static double GibbsEvidence<TDist>(IList<TDist> Uses, T Def)
            where TDist : IDistribution<T>
        {
            return 0.0;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static T UsesGibbs([IsReturned] T def, int resultIndex, T result)
        {
            return def;
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T UsesGibbs<TDist>(TDist def, T sample, int resultIndex, T result)
            where TDist : IDistribution<T>
        {
            // This method must depend on Def, even though Def isn't used, in order to get the right triggers
            return sample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static TDist UsesGibbs<TDist>(
            [IgnoreDependency] ICollection<TDist> Uses,
            [IsReturned] TDist Def,
            int resultIndex, TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            if (Uses.Count > 1)
                throw new ArgumentException("Uses.Count > 1");
            result.SetTo(Def);
            return result;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        [Skip]
        public static TDist UsesGibbsInit<TDist>([IgnoreDependency] TDist Def, int resultIndex)
            where TDist : ICloneable
        {
            return (TDist)Def.Clone();
        }

        /// <summary />
        /// <param name="def">Incoming message from <c>Def</c>.</param>
        /// <param name="sample">Buffer <c>sample</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        public static T DefGibbs<TDist>(TDist def, [IsReturned] T sample)
            where TDist : IDistribution<T>
        {
            return sample;
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="TDist">The type of the distribution over the replicated variable.</typeparam>
        //[MultiplyAll]
        public static TDist DefGibbs<TDist>(
            [SkipIfAllUniform] IList<TDist> Uses,
            TDist result)
            where TDist : IDistribution<T>, Sampleable<T>, SettableTo<TDist>, SettableToProduct<TDist>, SettableToRatio<TDist>
        {
            result.SetToUniform();
            result = Distribution.SetToProductWithAll(result, Uses);
            return result;
        }
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Replicate{T}(T, int)" /></description></item><item><description><see cref="Factor.ReplicateWithMarginal{T}(T, int, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Replicate<>", Default = false)]
    [FactorMethod(typeof(Factor), "ReplicateWithMarginal<>", Default = false)]
    [Quality(QualityBand.Mature)]
    public static class ReplicateOp_NoDivide
    {
        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] IList<T> Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }

#if SpecializeArrays
        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        [MultiplyAll]
        public static T MarginalAverageConditional<T>([NoInit] T[] Uses, T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            result.SetTo(Def);
            return Distribution.SetToProductWithAll(result, Uses);
        }
#endif

        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,Count)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] IList<T> Uses, T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Count)
                throw new ArgumentOutOfRangeException("resultIndex");
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }

#if SpecializeArrays
        /// <summary>EP message to <c>Uses</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[p(Uses) sum_(Def) p(Def) factor(Uses,Def,Count)]/p(Uses)</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [SkipIfAllUniform]
        public static T UsesAverageConditional<T>([AllExceptIndex] T[] Uses, T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            if (resultIndex < 0 || resultIndex >= Uses.Length)
                throw new ArgumentOutOfRangeException("resultIndex");
            result.SetTo(Def);
            return Distribution.SetToProductWithAllExcept(result, Uses, resultIndex);
        }
#endif

#if MinimalGenericTypeParameters
        /// <summary>EP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Def</c> as the random arguments are varied. The formula is <c>proj[p(Def) sum_(Uses) p(Uses) factor(Uses,Def,Count)]/p(Def)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }

#if SpecializeArrays
        /// <summary>EP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Def</c> as the random arguments are varied. The formula is <c>proj[p(Def) sum_(Uses) p(Uses) factor(Uses,Def,Count)]/p(Def)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageConditional<T>([SkipIfAllUniform] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#else
			public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] IList<TUses> Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#if SpecializeArrays
        public static T DefAverageConditional<T,TUses>([SkipIfAllUniform] TUses[] Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return Distribution.SetToProductOfAll(result, Uses);
        }
#endif
#endif
    }

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Replicate{T}(T, int)" /></description></item><item><description><see cref="Factor.ReplicateWithMarginal{T}(T, int, out T)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Replicate<>", Default = true)]
    [FactorMethod(typeof(Factor), "ReplicateWithMarginal<>", Default = true)]
    [Quality(QualityBand.Mature)]
    public static class ReplicateOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Uses,Def,Count))</c>.</para>
        /// </remarks>
        [Skip]
        public static double LogAverageFactor()
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="to_Uses">Outgoing message to <c>Uses</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(Uses,Def) p(Uses,Def) factor(Uses,Def,Count) / sum_Uses p(Uses) messageTo(Uses))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        public static double LogEvidenceRatio<T>([SkipIfAllUniform] IList<T> Uses, T Def, [Fresh] IList<T> to_Uses)
            where T : CanGetLogAverageOf<T>, SettableToProduct<T>, SettableTo<T>, ICloneable, SettableToUniform
        {
            return UsesEqualDefOp.LogEvidenceRatio(Uses, Def, to_Uses);
        }

        //-- VMP ----------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(Uses,Def,Count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            // Deterministic variables send no evidence messages.
            return 0.0;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the outgoing message.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        public static T MarginalAverageLogarithm<T, TDef>([SkipIfAllUniform] TDef Def, T result)
            where T : SettableTo<TDef>
        {
            return UsesAverageLogarithm<T, TDef>(Def, 0, result);
        }

        /// <summary>VMP message to <c>Uses</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>Uses</c> as the random arguments are varied. The formula is <c>proj[sum_(Def) p(Def) factor(Uses,Def,Count)]</c>.</para>
        /// </remarks>
        /// <typeparam name="T">The type of the outgoing message.</typeparam>
        /// <typeparam name="TDef">The type of the incoming message from <c>Def</c>.</typeparam>
        public static T UsesAverageLogarithm<T, TDef>([IsReturned] TDef Def, int resultIndex, T result)
            where T : SettableTo<TDef>
        {
            result.SetTo(Def);
            return result;
        }

        [Skip]
        public static T UsesDeriv<T>(T result)
            where T : SettableToUniform
        {
            result.SetToUniform();
            return result;
        }

        public static T UsesAverageLogarithm2<T, TDef>([IsReturnedInEveryElement] TDef Def, T result)
            where T : CanSetAllElementsTo<TDef>
        {
            result.SetAllElementsTo(Def);
            return result;
        }

        /// <summary>Initialize the buffer <c>Uses</c>.</summary>
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="count">Constant value for <c>Count</c>.</param>
        /// <param name="factory" />
        /// <returns>Initial value of buffer <c>Uses</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the incoming message from <c>Def</c>.</typeparam>
        /// <typeparam name="ArrayType">The type of arrays produced by <paramref name="factory"/>.</typeparam>
        [Skip]
        public static ArrayType UsesInit<T, ArrayType>([IgnoreDependency] T Def, int count, IArrayFactory<T, ArrayType> factory)
            where T : ICloneable
        {
            return factory.CreateArray(count, i => (T)Def.Clone());
        }

#if MinimalGenericTypeParameters
        /// <summary>VMP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Def</c> with <c>Uses</c> integrated out. The formula is <c>sum_Uses p(Uses) factor(Uses,Def,Count)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        [MultiplyAll]
        public static T DefAverageLogarithm<T>([SkipIfAllUniform, Trigger] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }

#if SpecializeArrays
        /// <summary>VMP message to <c>Def</c>.</summary>
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>Def</c> with <c>Uses</c> integrated out. The formula is <c>sum_Uses p(Uses) factor(Uses,Def,Count)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the messages.</typeparam>
        // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
        [MultiplyAll]
        public static T DefAverageLogarithm<T>([SkipIfAllUniform, Trigger] T[] Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return ReplicateOp_NoDivide.DefAverageConditional(Uses, result);
        }
#endif
#else
    // must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
        public static T DefAverageLogarithm<T,TUses>([SkipIfAllUniform,Trigger] IList<TUses> Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return DefAverageConditional(Uses, result);
        }
#if SpecializeArrays
			// must have upward Trigger to match the Trigger on UsesEqualDef.UsesAverageLogarithm
        public static T DefAverageLogarithm<T,TUses>([SkipIfAllUniform,Trigger] TUses[] Uses, T result)
            where T : SettableToProduct<TUses>, SettableTo<TUses>, TUses, SettableToUniform
        {
            return DefAverageConditional(Uses, result);
        }
#endif
#endif
    }

    /// <summary>Provides outgoing messages for <see cref="Factor.Replicate{T}(T, int)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Replicate<>")]
    [Quality(QualityBand.Mature)]
    public static class ReplicateMaxOp
    {
        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution o0ver the replicated variable.</typeparam>
        public static T UsesMaxConditional<T>([AllExceptIndex] IList<T> Uses, [SkipIfUniform] T Def, int resultIndex, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = ReplicateOp_NoDivide.UsesAverageConditional<T>(Uses, Def, resultIndex, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }

        /// <summary />
        /// <param name="Def">Incoming message from <c>Def</c>.</param>
        /// <param name="resultIndex">Index of the <c>Uses</c> for which a message is desired.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <typeparam name="T">The type of the distribution o0ver the replicated variable.</typeparam>
        [Skip]
        public static T UsesMaxConditionalInit<T>([IgnoreDependency] T Def, int resultIndex)
            where T : ICloneable
        {
            return (T)Def.Clone();
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>. Must be a proper distribution. If all elements are uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Uses" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution o0ver the replicated variable.</typeparam>
        public static T DefMaxConditional<T>([SkipIfAllUniform] IList<T> Uses, T result)
            where T : SettableToProduct<T>, SettableTo<T>, SettableToUniform
        {
            return ReplicateOp_NoDivide.DefAverageConditional<T>(Uses, result);
        }

        /// <summary />
        /// <param name="Uses">Incoming message from <c>Uses</c>.</param>
        /// <param name="Def">Incoming message from <c>Def</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Def" /> is not a proper distribution.</exception>
        /// <typeparam name="T">The type of the distribution over the replicated variable.</typeparam>
        public static T MarginalMaxConditional<T>(IList<T> Uses, [SkipIfUniform] T Def, T result)
            where T : SettableToProduct<T>, SettableTo<T>
        {
            T res = ReplicateOp_NoDivide.MarginalAverageConditional<T>(Uses, Def, result);
            if (res is UnnormalizedDiscrete)
                ((UnnormalizedDiscrete)(object)res).SetMaxToZero();
            return res;
        }
    }
}
