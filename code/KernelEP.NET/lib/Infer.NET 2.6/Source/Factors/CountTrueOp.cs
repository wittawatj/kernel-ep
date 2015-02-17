/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.CountTrue(bool[])" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "CountTrue")]
    [Quality(QualityBand.Preview)]
    [Buffers("PoissonBinomialTable")]
    public static class CountTrueOp
    {
        /// <summary>Initialize the buffer <c>PoissonBinomialTable</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>Initial value of buffer <c>PoissonBinomialTable</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double[,] PoissonBinomialTableInit(IList<Bernoulli> array)
        {
            return PoissonBinomialTable(array);
        }

        /// <summary>Update the buffer <c>PoissonBinomialTable</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>New value of buffer <c>PoissonBinomialTable</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        public static double[,] PoissonBinomialTable(IList<Bernoulli> array)
        {
            return PoissonBinomialForwardPass(array);
        }

        /// <summary>EP message to <c>count</c>.</summary>
        /// <param name="poissonBinomialTable">Buffer <c>poissonBinomialTable</c>.</param>
        /// <returns>The outgoing EP message to the <c>count</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>count</c> conditioned on the given values.</para>
        /// </remarks>
        /// <remarks><para>
        /// Marginal distribution of count is known as Poisson Binomial.
        /// It can be found in O(n^2) time using dynamic programming, where n is the length of the array.
        /// </para></remarks>
        public static Discrete CountAverageConditional([Fresh] double[,] poissonBinomialTable)
        {
            int tableSize = poissonBinomialTable.GetLength(0);
            return new Discrete(Util.ArrayInit(tableSize, i => poissonBinomialTable[tableSize - 1, i]));
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="count">Incoming message from <c>count</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="poissonBinomialTable">Buffer <c>poissonBinomialTable</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(count) p(count) factor(count,array)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="count" /> is not a proper distribution.</exception>
        /// <remarks><para>
        /// Poison Binomial for a given list of Bernoulli random variables with one variable excluded can be computed
        /// in linear time given Poisson Binomial for the whole list of items computed in a forward pass, as well as a special table
        /// containing averaged Poisson Binomial, which can be computed in a backward pass.
        /// Both tables can be computed in O(n^2) time, where n is a size of the list, so the time complexity of this message operator
        /// is also O(n^2).
        /// </para></remarks>
        /// <typeparam name="TBernoulliArray">The type of messages from/to 'array'.</typeparam>
        public static TBernoulliArray ArrayAverageConditional<TBernoulliArray>(
            TBernoulliArray array, [SkipIfUniform] Discrete count, [Fresh] double[,] poissonBinomialTable, TBernoulliArray result)
            where TBernoulliArray : IList<Bernoulli>
        {
            int i = array.Count - 1;
            foreach (double[] backwardPassTableRow in AveragedPoissonBinomialBackwardPassTableRows(array, count))
            {
                double probTrue = 0, probFalse = 0;
                for (int j = 0; j <= i; ++j)
                {
                    probTrue += poissonBinomialTable[i, j] * backwardPassTableRow[j + 1];
                    probFalse += poissonBinomialTable[i, j] * backwardPassTableRow[j];
                }

                Debug.Assert(probTrue + probFalse > 1e-10, "The resulting distribution should be well-defined.");
                result[i] = new Bernoulli(probTrue / (probFalse + probTrue));

                --i;
            }

            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="count">Constant value for <c>count</c>.</param>
        /// <param name="poissonBinomialTable">Buffer <c>poissonBinomialTable</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TBernoulliArray">The type of messages from/to 'array'.</typeparam>
        public static TBernoulliArray ArrayAverageConditional<TBernoulliArray>(
            TBernoulliArray array, int count, double[,] poissonBinomialTable, TBernoulliArray result)
            where TBernoulliArray : IList<Bernoulli>
        {
            Discrete mass = Discrete.PointMass(count, array.Count + 1);
            return ArrayAverageConditional(array, mass, poissonBinomialTable, result);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="count">Incoming message from <c>count</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(count) p(count) factor(count,array) / sum_count p(count) messageTo(count))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="count" /> is not a proper distribution.</exception>
        [Skip]
        public static double LogEvidenceRatio([SkipIfUniform] Discrete count)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="count">Constant value for <c>count</c>.</param>
        /// <param name="poissonBinomialTable">Buffer <c>poissonBinomialTable</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(count,array))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int count, [Fresh] double[,] poissonBinomialTable)
        {
            int tableSize = poissonBinomialTable.GetLength(0);
            return Math.Log(poissonBinomialTable[tableSize - 1, count]);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="count">Constant value for <c>count</c>.</param>
        /// <param name="array">Constant value for <c>array</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(count,array))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(int count, bool[] array)
        {
            return (count == Factor.CountTrue(array)) ? 0.0 : double.NegativeInfinity;
        }

        #region Helpers

        /// <summary>
        /// Compute Poisson Binomial table for a given list of Bernoulli-distributed random variables in a forward pass.
        /// </summary>
        /// <param name="array">List of Bernoulli random variables.</param>
        /// <returns>Table A, such that A[i, j] = P(sum(<paramref name="array"/>[1], ..., <paramref name="array"/>[i]) = j).</returns>
        private static double[,] PoissonBinomialForwardPass(IList<Bernoulli> array)
        {
            var result = new double[array.Count + 1, array.Count + 1];

            result[0, 0] = 1.0;
            for (int i = 1; i <= array.Count; ++i)
            {
                double probTrue = array[i - 1].GetProbTrue();
                result[i, 0] = result[i - 1, 0] * (1 - probTrue);
                for (int j = 1; j <= i; ++j)
                {
                    result[i, j] = (result[i - 1, j] * (1 - probTrue)) + (result[i - 1, j - 1] * probTrue);
                }
            }

            return result;
        }

        /// <summary>
        /// Enumerate rows of averaged Poisson Binomial table for a given list of Bernoulli-distributed random variables in a backward pass.
        /// </summary>
        /// <param name="array">List of Bernoulli random variables.</param>
        /// <param name="averager">Distribution over sum of values in <paramref name="array"/> used to average the Poisson Binomial.</param>
        /// <returns>
        /// <para>
        /// Rows of table A, such that A[i, j] =
        /// \sum_c P(<paramref name="averager"/> = c) P(sum(<paramref name="array"/>[i + 1], ..., <paramref name="array"/>[n]) + j = c).
        /// Rows are returned last-to-first, the very first row is omitted.
        /// </para>
        /// </returns>
        private static IEnumerable<double[]> AveragedPoissonBinomialBackwardPassTableRows(IList<Bernoulli> array, Discrete averager)
        {
            Debug.Assert(averager.Dimension == array.Count + 1, "'averager' should represent a distribution over the sum of the elements of 'array'.");
            var prevRow = new double[array.Count + 1];
            var currentRow = new double[array.Count + 1];
            for (int j = 0; j <= array.Count; ++j)
            {
                currentRow[j] = averager[j];
            }

            for (int i = array.Count - 1; i >= 0; --i)
            {
                yield return currentRow;
                double[] temp = currentRow;
                currentRow = prevRow;
                prevRow = temp;

                double probTrue = array[i].GetProbTrue();
                currentRow[array.Count] = prevRow[array.Count] * (1 - probTrue);
                for (int j = array.Count - 1; j >= 0; --j)
                {
                    currentRow[j] = (prevRow[j] * (1 - probTrue)) + (prevRow[j + 1] * probTrue);
                }
            }
        }

        #endregion
    }
}
