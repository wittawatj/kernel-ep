/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Factors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Sum(IList{Vector})" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Sum", typeof(Vector[]))]
    [Quality(QualityBand.Stable)]
    public static class SumVectorGaussianOp
    {
        #region EP

        #region Forward messages

        /// <summary>EP message to <c>sum</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sum</c> as the random arguments are varied. The formula is <c>proj[p(sum) sum_(array) p(array) factor(sum,array)]/p(sum)</c>.</para>
        /// </remarks>
        public static VectorGaussian SumAverageConditional([SkipIfAnyUniform] IList<VectorGaussian> array, VectorGaussian result)
        {
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            if (result == null)
            {
                throw new ArgumentNullException("result");
            }

            if (array.Count < 1)
            {
                result.Point = Vector.Zero(result.Dimension);
                return result;
            }

            if (array.Any(element => element == null))
            {
                throw new ArgumentNullException("array");
            }

            int dimension = result.Dimension;
            if (array.Any(element => element.Dimension != dimension))
            {
                throw new ArgumentException("The result and all elements of the array must have the same number of dimensions.");
            }

            var sumMean = Vector.Zero(dimension);
            var sumVariance = PositiveDefiniteMatrix.IdentityScaledBy(dimension, 0);
            var elementMean = Vector.Zero(dimension);
            var elementVariance = PositiveDefiniteMatrix.Identity(dimension);

            foreach (var element in array)
            {
                if (!element.IsProper())
                {
                    return element;
                }

                element.GetMeanAndVariance(elementMean, elementVariance);

                sumMean.SetToSum(sumMean, elementMean);
                sumVariance.SetToSum(sumVariance, elementVariance);
            }

            result.SetMeanAndVariance(sumMean, sumVariance);

            return result;
        }

        #endregion

        #region Backward messages

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="sum">Incoming message from <c>sum</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_sum">Outgoing message to <c>sum</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(sum) p(sum) factor(sum,array)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        public static TVectorGaussianList ArrayAverageConditional<TVectorGaussianList>(
            [SkipIfUniform] VectorGaussian sum,
            [Fresh] VectorGaussian to_sum,
            IList<VectorGaussian> array,
            TVectorGaussianList result)
            where TVectorGaussianList : IList<VectorGaussian>, SettableToUniform
        {
            // Check inputs for consistency
            int dimension = CheckArgumentConsistency(sum, to_sum, array, result);

            if (array.Count == 0)
            {
                return result;
            }

            // It is tempting to put SkipIfAllUniform on array but this isn't correct if the array has one element
            if (array.Count == 1)
            {
                result[0].SetTo(sum);
                return result;
            }

            if (!sum.IsProper())
            {
                foreach (VectorGaussian element in result)
                {
                    element.SetTo(sum);
                }

                return result;
            }

            var elementMean = Vector.Zero(dimension);
            var elementVariance = PositiveDefiniteMatrix.Identity(dimension);

            // Check if an element of the array is uniform
            int indexOfUniform = -1;
            for (int i = 0; i < array.Count; i++)
            {
                array[i].GetMeanAndVariance(elementMean, elementVariance);

                // Instead of testing IsUniform, we need to test the more strict requirement that all diagonal 
                // elements of variance are infinite due to the way we are doing the computations
                if (IsUniform(elementVariance))
                {
                    if (indexOfUniform >= 0)
                    {
                        // More than one element of array is uniform
                        result.SetToUniform();
                        return result;
                    }

                    indexOfUniform = i;
                }
            }

            Vector sumMean = Vector.Zero(dimension);
            PositiveDefiniteMatrix sumVariance = PositiveDefiniteMatrix.Identity(dimension);
            sum.GetMeanAndVariance(sumMean, sumVariance);

            Vector totalMean = Vector.Zero(sum.Dimension);
            PositiveDefiniteMatrix totalVariance = PositiveDefiniteMatrix.IdentityScaledBy(sum.Dimension, 0.0);

            if (indexOfUniform >= 0)
            {
                // Exactly one element of array is uniform
                for (int i = 0; i < array.Count; i++)
                {
                    if (i == indexOfUniform)
                    {
                        continue;
                    }

                    array[i].GetMeanAndVariance(elementMean, elementVariance);
                    totalMean.SetToSum(totalMean, elementMean);
                    totalVariance.SetToSum(totalVariance, elementVariance);
                    result[i].SetToUniform();
                }

                // totalMean = sum_{i except indexOfUniform} array[i].GetMean()
                // totalVariance = sum_{i except indexOfUniform} array[i].GetVariance()
                totalMean.SetToDifference(sumMean, totalMean);
                totalVariance.SetToSum(sumVariance, totalVariance);
                result[indexOfUniform].SetMeanAndVariance(totalMean, totalVariance);
                return result;
            }

            // At this point, the array has no uniform elements

            // Get the mean and variance of sum of all Gaussians
            to_sum.GetMeanAndVariance(totalMean, totalVariance);

            // Subtract it off from the mean and variance of the incoming Gaussian from Sum
            totalMean.SetToDifference(sumMean, totalMean);
            totalVariance.SetToSum(totalVariance, sumVariance);

            for (int i = 0; i < array.Count; i++)
            {
                array[i].GetMeanAndVariance(elementMean, elementVariance);
                elementMean.SetToSum(elementMean, totalMean);
                elementVariance.SetToDifference(totalVariance, elementVariance);
                result[i].SetMeanAndVariance(elementMean, elementVariance);
            }

            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="sum">Incoming message from <c>sum</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_sum">Outgoing message to <c>sum</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>array</c> as the random arguments are varied. The formula is <c>proj[p(array) sum_(sum) p(sum) factor(sum,array)]/p(array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        public static VectorGaussian[] ArrayAverageConditional(
            [SkipIfUniform] VectorGaussian sum,
            [Fresh] VectorGaussian to_sum,
            VectorGaussian[] array,
            VectorGaussian[] result)
        {
            // Check inputs for consistency
            int dimension = CheckArgumentConsistency(sum, to_sum, array, result);

            if (array.Length == 0)
            {
                return result;
            }

            // It is tempting to put SkipIfAllUniform on array but this isn't correct if the array has one element
            if (array.Length == 1)
            {
                result[0] = sum;
                return result;
            }

            if (!sum.IsProper())
            {
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = sum;
                }

                return result;
            }

            var elementMean = Vector.Zero(dimension);
            var elementVariance = PositiveDefiniteMatrix.Identity(dimension);

            // Check if an element of the array is uniform
            int indexOfUniform = -1;
            for (int i = 0; i < array.Length; i++)
            {
                array[i].GetMeanAndVariance(elementMean, elementVariance);

                // Instead of testing IsUniform, we need to test the more strict requirement that all diagonal 
                // elements of variance are infinite due to the way we are doing the computations
                if (IsUniform(elementVariance))
                {
                    if (indexOfUniform >= 0)
                    {
                        // More than one element of array is uniform
                        foreach (var element in result)
                        {
                            element.SetToUniform();
                        }

                        return result;
                    }

                    indexOfUniform = i;
                }
            }

            var sumMean = Vector.Zero(dimension);
            var sumVariance = PositiveDefiniteMatrix.Identity(dimension);
            sum.GetMeanAndVariance(sumMean, sumVariance);

            var totalMean = Vector.Zero(sum.Dimension);
            var totalVariance = PositiveDefiniteMatrix.IdentityScaledBy(sum.Dimension, 0.0);

            if (indexOfUniform >= 0)
            {
                // Exactly one element of array is uniform
                for (int i = 0; i < array.Length; i++)
                {
                    if (i == indexOfUniform)
                    {
                        continue;
                    }

                    array[i].GetMeanAndVariance(elementMean, elementVariance);
                    totalMean.SetToSum(totalMean, elementMean);
                    totalVariance.SetToSum(totalVariance, elementVariance);
                    result[i].SetToUniform();
                }

                // totalMean = sum_{i except indexOfUniform} array[i].GetMean()
                // totalVariance = sum_{i except indexOfUniform} array[i].GetVariance()
                totalMean.SetToDifference(sumMean, totalMean);
                totalVariance.SetToSum(totalVariance, sumVariance);
                result[indexOfUniform].SetMeanAndVariance(totalMean, totalVariance);
                return result;
            }

            // At this point, the array has no uniform elements

            // Get the mean and variance of sum of all the Gaussians
            to_sum.GetMeanAndVariance(totalMean, totalVariance);

            // Subtract it off from the mean and variance of the incoming Gaussian from Sum
            totalMean.SetToDifference(sumMean, totalMean);
            totalVariance.SetToSum(totalVariance, sumVariance);

            for (int i = 0; i < array.Length; i++)
            {
                array[i].GetMeanAndVariance(elementMean, elementVariance);
                elementMean.SetToSum(elementMean, totalMean);
                elementVariance.SetToDifference(totalVariance, elementVariance);
                result[i].SetMeanAndVariance(elementMean, elementVariance);
            }

            return result;
        }

        /// <summary>EP message to <c>array</c>.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> conditioned on the given values.</para>
        /// </remarks>
        /// <typeparam name="TVectorGaussianList">A list of <see cref="VectorGaussian"/> distributions which may be set to uniform.</typeparam>
        public static TVectorGaussianList ArrayAverageConditional<TVectorGaussianList>(
            [SkipIfUniform] Vector sum,
            IList<VectorGaussian> array,
            TVectorGaussianList result)
            where TVectorGaussianList : IList<VectorGaussian>, SettableToUniform
        {
            if (sum == null)
            {
                throw new ArgumentNullException("sum");
            }

            VectorGaussian to_sum = SumAverageConditional(array, new VectorGaussian(sum.Count));
            return ArrayAverageConditional(VectorGaussian.PointMass(sum), to_sum, array, result);
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(array) p(array) factor(sum,array))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector sum, IList<VectorGaussian> array)
        {
            return LogAverageFactor(sum, array);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Constant value for <c>array</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sum,array))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector sum, IList<Vector> array)
        {
            return LogAverageFactor(sum, array);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Incoming message from <c>sum</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sum) p(sum) factor(sum,array) / sum_sum p(sum) messageTo(sum))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian sum)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(array) p(array) factor(sum,array))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector sum, [SkipIfAnyUniform] IList<VectorGaussian> array)
        {
            if (sum == null)
            {
                throw new ArgumentNullException("sum");
            }

            VectorGaussian to_sum = SumAverageConditional(array, new VectorGaussian(sum.Count));
            return to_sum.GetLogProb(sum);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Constant value for <c>array</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sum,array))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector sum, IList<Vector> array)
        {
            if (sum == null)
            {
                throw new ArgumentNullException("sum");
            }

            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            if (array.Count == 0)
            {
                return (sum == Vector.Zero(sum.Count)) ? 0.0 : double.NegativeInfinity;
            }

            return (sum == Factor.Sum(array)) ? 0.0 : double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="sum">Incoming message from <c>sum</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_sum">Outgoing message to <c>sum</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(sum) p(sum) factor(sum,array))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        public static double LogAverageFactor([SkipIfUniform] VectorGaussian sum, [Fresh] VectorGaussian to_sum)
        {
            return to_sum.GetLogAverageOf(sum);
        }

        #endregion

        #endregion

        #region VMP

        #region Forward messages

        /// <summary>VMP message to <c>sum</c>.</summary>
        /// <param name="array">Incoming message from <c>array</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>sum</c> as the random arguments are varied. The formula is <c>proj[sum_(array) p(array) factor(sum,array)]</c>.</para>
        /// </remarks>
        public static VectorGaussian SumAverageLogarithm([SkipIfAnyUniform] IList<VectorGaussian> array, VectorGaussian result)
        {
            return SumAverageConditional(array, result);
        }

        #endregion

        #region Backward messages

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="sum">Incoming message from <c>sum</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_array">Previous outgoing message to <c>array</c>.</param>
        /// <returns>The outgoing VMP message to the <c>array</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> with <c>sum</c> integrated out. The formula is <c>sum_sum p(sum) factor(sum,array)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="TVectorGaussianList">A list of <see cref="VectorGaussian"/> distributions.</typeparam>
        public static TVectorGaussianList ArrayAverageLogarithm<TVectorGaussianList>(
            [SkipIfUniform] VectorGaussian sum,
            [Proper] IList<VectorGaussian> array,
            TVectorGaussianList to_array)
            where TVectorGaussianList : IList<VectorGaussian>, ICloneable
        {
            return ArrayAverageLogarithm1(sum, array, to_array);
        }

        /// <summary />
        /// <param name="sum">Incoming message from <c>sum</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="to_array">Previous outgoing message to <c>array</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="sum" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="TVectorGaussianList">A list of <see cref="VectorGaussian"/> distributions.</typeparam>
        public static TVectorGaussianList ArrayAverageLogarithm1<TVectorGaussianList>(
            [SkipIfUniform] VectorGaussian sum,
            [Stochastic, Proper] IList<VectorGaussian> array,
            TVectorGaussianList to_array)
            where TVectorGaussianList : IList<VectorGaussian>
        {
            // Check inputs for consistency
            int dimension = CheckArgumentConsistency(sum, sum, array, to_array);

            TVectorGaussianList result = to_array;

            var sumMean = Vector.Zero(dimension);
            var sumVariance = PositiveDefiniteMatrix.Identity(dimension);
            sum.GetMeanAndVariance(sumMean, sumVariance);

            // This version does one update of q(array[i]) for each array element in turn.
            Vector arraySumOfMean = Vector.Zero(dimension);
            foreach (VectorGaussian element in array)
            {
                arraySumOfMean.SetToSum(arraySumOfMean, element.GetMean());
            }

            for (int i = 0; i < result.Count; i++)
            {
                arraySumOfMean.SetToDifference(arraySumOfMean, array[i].GetMean());

                VectorGaussian oldResult = result[i];
                result[i] = new VectorGaussian(sumMean - arraySumOfMean, sumVariance);

                oldResult.SetToRatio(result[i], oldResult);
                oldResult.SetToProduct(array[i], oldResult);

                arraySumOfMean.SetToSum(arraySumOfMean, oldResult.GetMean());
            }

            return result;
        }

        /// <summary>VMP message to <c>array</c>.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Incoming message from <c>array</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>array</c> conditioned on the given values.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="array" /> is not a proper distribution.</exception>
        /// <typeparam name="TVectorGaussianList">A list of <see cref="VectorGaussian"/> distributions.</typeparam>
        public static TVectorGaussianList ArrayAverageLogarithm<TVectorGaussianList>(
            Vector sum, [Proper] IList<VectorGaussian> array, TVectorGaussianList result)
            where TVectorGaussianList : IList<VectorGaussian>, ICloneable
        {
            return ArrayAverageLogarithm(VectorGaussian.PointMass(sum), array, result);
        }

        #endregion

        #region Evidence messages

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="sum">Constant value for <c>sum</c>.</param>
        /// <param name="array">Constant value for <c>array</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sum,array))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Vector sum, IList<Vector> array)
        {
            return LogAverageFactor(sum, array);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(sum,array))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor()
        {
            return 0.0;
        }

        #endregion

        #endregion

        #region Helper

        /// <summary>
        /// Tests whether all elements on the diagonal of variance are infinite.
        /// </summary>
        /// <param name="variance">The variance, a <see cref="PositiveDefiniteMatrix"/>.</param>
        /// <returns>True, if all elements on the diagonal of <paramref name="variance"/> are infinite.</returns>
        private static bool IsUniform(PositiveDefiniteMatrix variance)
        {
            bool isUniform = true;
            for (int d = 0; d < variance.Rows; d++)
            {
                if (double.IsPositiveInfinity(variance[d, d]))
                {
                    continue;
                }

                isUniform = false;
                break;
            }

            return isUniform;
        }

        /// <summary>
        /// Checks arguments for consistency and returns the dimension of the arguments' distributions.
        /// </summary>
        /// <param name="sum">Incoming message from 'sum'.</param>
        /// <param name="to_sum">Outgoing message to 'sum'.</param>
        /// <param name="array">Incoming message from 'array'.</param>
        /// <param name="result">Contains the outgoing message.</param>
        /// <returns>The dimension of the distributions.</returns>
        /// <exception cref="ArgumentNullException">Thrown when sum, array, or any element of array is null.</exception>
        /// <exception cref="ArgumentException">Thrown for inconsistent arguments.</exception>
        private static int CheckArgumentConsistency(VectorGaussian sum, VectorGaussian to_sum, ICollection<VectorGaussian> array, ICollection<VectorGaussian> result)
        {
            if (sum == null)
            {
                throw new ArgumentNullException("sum");
            }

            if (to_sum == null)
            {
                throw new ArgumentNullException("to_sum");
            }

            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            if (result == null)
            {
                throw new ArgumentNullException("result");
            }

            if (array.Count != result.Count)
            {
                throw new ArgumentException("The array and the result must have the same length, but are " + array.Count + " and " + result.Count + " respectively.");
            }

            if (array.Any(element => element == null))
            {
                throw new ArgumentNullException("array");
            }

            if (result.Any(element => element == null))
            {
                throw new ArgumentNullException("result");
            }

            int dimension = sum.Dimension;
            if (array.Any(element => element.Dimension != dimension))
            {
                throw new ArgumentException("The distribution of sum and all distributions in the array must have the same number of dimensions.");
            }

            if (result.Any(element => element.Dimension != dimension))
            {
                throw new ArgumentException("The distribution of sum and all distributions in result must have the same number of dimensions.");
            }

            if (to_sum.Dimension != dimension)
            {
                throw new ArgumentException("The incoming message from sum and the outgoing message to sum must have the same number of dimensions.");
            }

            return dimension;
        }

        #endregion
    }
}
