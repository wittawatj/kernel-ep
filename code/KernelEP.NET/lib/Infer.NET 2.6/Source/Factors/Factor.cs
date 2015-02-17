// (C) Copyright 2008 Microsoft Research Cambridge

using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Distributions.Kernels;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;

namespace MicrosoftResearch.Infer.Factors
{
    using System.Diagnostics;

    /// <summary>
    /// A repository of commonly used factor methods.
    /// </summary>
    [Quality(QualityBand.Stable)]
    public static class Factor
    {
        /// <summary>
        /// Random factor - samples from a given distribution
        /// </summary>
        /// <typeparam name="DomainType">Domain type</typeparam>
        /// <param name="dist">Distribution to sample from</param>
        /// <returns>Sample</returns>
        [Stochastic]
        [Hidden]
        public static DomainType Random<DomainType>(Sampleable<DomainType> dist)
        {
            return dist.Sample();
        }

        /// <summary>
        /// Sample from a Bernoulli distribution.
        /// </summary>
        /// <param name="probTrue">The probability that the result is true.</param>
        /// <returns>A random boolean value.</returns>
        [Stochastic]
        [ParameterNames("sample", "probTrue")]
        public static bool Bernoulli(double probTrue)
        {
            return MicrosoftResearch.Infer.Distributions.Bernoulli.Sample(probTrue);
        }

        /// <summary>
        /// Sample from a Bernoulli distribution with specified log odds
        /// </summary>
        /// <param name="logOdds"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("sample", "logOdds")]
        public static bool BernoulliFromLogOdds(double logOdds)
        {
            return Bernoulli(MMath.Logistic(logOdds));
        }

        /// <summary>
        /// Sample from a Gamma with specified shape and scale
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="rate"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("sample", "shape", "rate")]
        public static double GammaFromShapeAndRate(double shape, double rate)
        {
            return Gamma.Sample(shape, 1/rate);
        }

        /// <summary>
        /// Sample from one of many Bernoulli distributions. This factor is DEPRECATED.
        /// Use Gates instead.
        /// </summary>
        /// <param name="index">The index of the distribution to sample from.</param>
        /// <param name="probTrue">The probability that the result is true, for each index.</param>
        /// <returns>A random boolean value.</returns>
        /// <exclude/>
        [Stochastic]
        [ParameterNames("sample", "index", "probTrue")]
        [System.Obsolete("Use Gates instead")]
        public static bool BernoulliFromDiscrete(int index, params double[] probTrue)
        {
            return Bernoulli(probTrue[index]);
        }

        /// <summary>
        /// Sample from one of two Bernoulli distributions. This factor is DEPRECATED.
        /// Use Gates instead.
        /// </summary>
        /// <param name="choice">Indicates which distribution to sample from.</param>
        /// <param name="probTrue">The probability that the result is true, for each choice.</param>
        /// <returns>A random boolean value.</returns>
        /// <exclude/>
        [Stochastic]
        [ParameterNames("sample", "choice", "probTrue")]
        [System.Obsolete("Use Gates instead")]
        public static bool BernoulliFromBoolean(bool choice, params double[] probTrue)
        {
            return Bernoulli(choice ? probTrue[1] : probTrue[0]);
        }

        /// <summary>
        /// Sample from one of two Bernoulli distributions. This factor is DEPRECATED.
        /// Use Gates instead.
        /// </summary>
        /// <param name="choice">Indicates which distribution to sample from.</param>
        /// <param name="probTrueElseChoice">The probability that the result is true, if <paramref name="choice"/> is false.</param>
        /// <param name="probTrueIfChoice">The probability that the result is true, if <paramref name="choice"/> is true.</param>
        /// <returns>A random boolean value.</returns>
        /// <exclude/>
        [System.Obsolete("Use Gates instead")]
        [Stochastic]
        [ParameterNames("sample", "choice", "probTrue0", "probTrue1")]
        public static bool BernoulliFromBoolean(bool choice, double probTrueElseChoice, double probTrueIfChoice)
        {
            return Bernoulli(choice ? probTrueIfChoice : probTrueElseChoice);
        }

        /// <summary>
        /// Sample from a Beta distribution
        /// </summary>
        /// <param name="mean">Mean of the distribution.</param>
        /// <param name="totalCount">Total count (precision) of the distribution.</param>
        /// <returns>A sample from the distribution, i.e. a value in [0,1]. </returns>
        [Stochastic]
        [ParameterNames("prob", "mean", "totalCount")]
        public static double BetaFromMeanAndTotalCount(double mean, double totalCount)
        {
            return Rand.Beta(mean*totalCount, (1 - mean)*totalCount);
        }

        /// <summary>
        /// Sample from a Dirichlet distribution
        /// </summary>
        /// <param name="mean">Mean of the distribution.</param>
        /// <param name="totalCount">Total count (precision) of the distribution.</param>
        /// <returns>A sample from the distribution, a probability vector. </returns>
        [Stochastic]
        [ParameterNames("prob", "mean", "totalCount")]
        public static Vector DirichletFromMeanAndTotalCount(Vector mean, double totalCount)
        {
            Vector v = Vector.Copy(mean);
            for (int i = 0; i < mean.Count; i++)
                v[i] = Rand.Gamma(mean[i]*totalCount);
            v.Scale(1/Math.Sqrt(v.Inner(v)));
            return v;
        }

        /// <summary>
        /// Sample from a symmetric Dirichlet distribution Dir(alpha,...,alpha)
        /// </summary>
        /// <param name="K">Dimension of the distribution.</param>
        /// <param name="alpha">The hyperparameter.</param>
        /// <returns>A sample from the distribution, a probability vector. </returns>
        [Stochastic]
        [ParameterNames("prob", "K", "alpha")]
        public static Vector DirichletSymmetric([Constant] int K, double alpha)
        {
            var v = Vector.Zero(K);
            for (int i = 0; i < K; i++)
                v[i] = Rand.Gamma(alpha);
            v.Scale(1/v.Sum());
            return v;
        }

        /// <summary>
        /// Sample from one of several discrete distributions.
        /// </summary>
        /// <param name="probs">Matrix holding discrete distributions as rows.</param>
        /// <param name="selector">Integer selecting which row of probs to sample from.</param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("sample", "selector", "probs")]
        [Hidden]
        public static int Discrete(int selector, Matrix probs)
        {
            return MicrosoftResearch.Infer.Distributions.Discrete.Sample(probs.RowVector(selector));
        }

        /// <summary>
        /// Sample from a discrete distribution.
        /// </summary>
        /// <param name="probs">The probability of each outcome.</param>
        /// <returns>A random integer from 0 to <c>probs.Length-1</c></returns>
        [Stochastic]
        [ParameterNames("sample", "probs")]
        public static int Discrete(Vector probs)
        {
            return MicrosoftResearch.Infer.Distributions.Discrete.Sample(probs);
        }

        /// <summary>
        /// Sample from a uniform discrete distribution
        /// </summary>
        /// <param name="size">The dimension of the distribution (how many possibke distinct values</param>
        /// <returns>A random integer from 0 to <c>size-1</c></returns>
        [Stochastic]
        [ParameterNames("sample", "size")]
        public static int DiscreteUniform(int size)
        {
            return Rand.Int(size);
        }

        /// <summary>
        /// Sample from a stick breaking distribution
        /// </summary>
        /// <param name="v">Stick lengths.  The last value is ignored.</param>
        /// <returns>A random integer in 0..(v.Length-1)</returns>
        [Stochastic]
        [ParameterNames("c", "v")]
        public static int DiscreteFromStickBreaking(double[] v)
        {
            for (int i = 0; i < v.Length-1; i++)
                if (Bernoulli(v[i]))
                    return i;
            return v.Length - 1;
        }

        /*
        /// <summary>
        /// Sample a discrete probability distribution from a Pitman-Yor(a,b) process
        /// </summary>
        /// <param name="a">Parameter a.</param>
        /// <param name="b">Parameter b.</param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("pitmanyor", "a", "b")]
        public static Vector PitmanYor(double a, double b)
        {
            int k = 5; // how do I get this?
            Vector p = new Vector(k);
            double remainingStick=1;
            for (int i = 0; i < k; i++)
            {
                double a_k = 1 - a;
                double b_k = b + (i + 1) * a;
                double v = MicrosoftResearch.Infer.Distributions.Beta.Sample(a_k, b_k);
                p[i] = remainingStick * v;
                remainingStick -= p[i];
            }
            p[k - 1] += remainingStick;
            return p;
        }*/

        /// <summary>
        /// Sample from a Gaussian distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="precision">The precision of the distribution.  The variance will be 1/precision.</param>
        /// <returns>A random real number.</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "precision")]
        public static double Gaussian(double mean, double precision)
        {
            return MicrosoftResearch.Infer.Distributions.Gaussian.Sample(mean, precision);
        }

        /// <summary>
        /// Sample from a Gaussian distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="variance">The variance of the distribution.  The precision will be 1/variance.</param>
        /// <returns>A random real number.</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "variance")]
        public static double GaussianFromMeanAndVariance(double mean, double variance)
        {
            return Rand.Normal(mean, Math.Sqrt(variance));
        }

        /// <summary>
        /// Sample from a VectorGaussian distribution.
        /// </summary>
        /// <param name="mean">The mean vector of the distribution.</param>
        /// <param name="precision">The precision matrix of the distribution.  The variance matrix will be inv(precision).</param>
        /// <returns>A random real vector.</returns>
        [Stochastic]
        [ParameterNames("sample", "mean", "precision")]
        public static Vector VectorGaussian(Vector mean, PositiveDefiniteMatrix precision)
        {
            return MicrosoftResearch.Infer.Distributions.VectorGaussian.Sample(mean, precision);
        }

        /// <summary>
        /// Sample from a Poisson distribution with a specified mean
        /// </summary>
        /// <param name="mean">The mean of the Poisson distribution</param>
        /// <returns>An integer sample &gt;= 0</returns>
        [Stochastic]
        [ParameterNames("sample", "mean")]
        public static int Poisson(double mean)
        {
            return MicrosoftResearch.Infer.Distributions.Poisson.Sample(mean);
        }

        /// <summary>
        /// Sample from a Poisson distribution with specified log mean
        /// </summary>
        /// <param name="logRate">The log mean of the Poisson distribution</param>
        /// <returns>An integer sample &gt;= 0</returns>
        [Stochastic]
        [ParameterNames("sample", "logRate")]
        public static int PoissonFromLogRate(double logRate)
        {
            return MicrosoftResearch.Infer.Distributions.Poisson.Sample(Math.Exp(logRate));
        }

        /// <summary>
        /// Sample from a Binomial distribution with specified probability of success per trial and number of trials.
        /// </summary>
        /// <param name="trialCount"></param>
        /// <param name="probSuccess"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("sample", "trialCount", "p")]
        public static int Binomial(int trialCount, double probSuccess)
        {
            return Rand.Binomial(trialCount, probSuccess);
        }

        /// <summary>
        /// Sample from a Multinomial distribution with specified probabilities and number of trials.
        /// </summary>
        /// <param name="trialCount">Number of trials, >= 0</param>
        /// <param name="probs">Must sum to 1</param>
        /// <returns>An array of length <c>probs.Count</c> of integers between 0 and trialCount, whose sum is trialCount.</returns>
        [Stochastic]
        [ParameterNames("sample", "trialCount", "p")]
        public static int[] Multinomial(int trialCount, Vector probs)
        {
            return Rand.Multinomial(trialCount, probs);
        }

        /// <summary>
        /// Sample from a Multinomial distribution with specified probabilities and number of trials.
        /// </summary>
        /// <param name="trialCount">Number of trials, >= 0</param>
        /// <param name="probs">Must sum to 1</param>
        /// <returns>A list of length <c>probs.Count</c> of integers between 0 and trialCount, whose sum is trialCount.</returns>
        [Stochastic]
        [ParameterNames("sample", "trialCount", "p")]
        public static IList<int> MultinomialList(int trialCount, Vector probs)
        {
            return Multinomial(trialCount, probs);
        }

        /// <summary>
        /// Negate a boolean
        /// </summary>
        /// <param name="b">The bool</param>
        /// <returns>The negation of a boolean argument</returns>
        public static bool Not(bool b)
        {
            return !b;
        }

        /// <summary>
        /// Logical or of two booleans
        /// </summary>
        /// <param name="a">First bool</param>
        /// <param name="b">Second bool</param>
        /// <returns>a|b</returns>
        public static bool Or(bool a, bool b)
        {
            return (a | b);
        }

        /// <summary>
        /// Logical and of two booleans
        /// </summary>
        /// <param name="a">First bool</param>
        /// <param name="b">Second bool</param>
        /// <returns>a&amp;b</returns>
        public static bool And(bool a, bool b)
        {
            return (a & b);
        }

        /// <summary>
        /// Test if two booleans are equal.
        /// </summary>
        /// <param name="a">First bool</param>
        /// <param name="b">Second bool</param>
        /// <returns>True if a==b.</returns>
        public static bool AreEqual(bool a, bool b)
        {
            return (a == b);
        }

        /// <summary>
        /// Test if two integers are equal.
        /// </summary>
        /// <param name="a">First integer</param>
        /// <param name="b">Second integer</param>
        /// <returns>True if a==b.</returns>
        public static bool AreEqual(int a, int b)
        {
            return (a == b);
        }

        /// <summary>
        /// Tests if two strings are equal.
        /// </summary>
        /// <param name="str1">The first string.</param>
        /// <param name="str2">The second string.</param>
        /// <returns><see langword="true"/> if the strings are equal, <see langword="false"/> otherwise.</returns>
        public static bool AreEqual(string str1, string str2)
        {
            return str1 == str2;
        }

        /// <summary>
        /// Test if a real number is positive.
        /// </summary>
        /// <param name="x">Any number besides NaN.</param>
        /// <returns>True if x>0.</returns>
        public static bool IsPositive(double x)
        {
            return (x > 0);
        }

        /// <summary>
        /// Test if a number is between two bounds.
        /// </summary>
        /// <param name="x">Any number besides NaN.</param>
        /// <param name="lowerBound">Any number besides NaN.</param>
        /// <param name="upperBound">Any number besides NaN.</param>
        /// <returns>True if (lowerBound &lt;= x) and (x &lt; upperBound)</returns>
        public static bool IsBetween(double x, double lowerBound, double upperBound)
        {
            return (lowerBound <= x) && (x < upperBound);
        }

        /// <summary>
        /// Returns the maximum of the two arguments: max(a,b)
        /// </summary>
        /// <returns>a*b</returns>
        public static double Max(double a, double b)
        {
            return Math.Max(a, b);
        }

        /// <summary>
        /// Test if A is greater than B.
        /// </summary>
        /// <param name="a">First integer</param>
        /// <param name="b">Second integer</param>
        /// <returns>True if a &gt; b</returns>
        public static bool IsGreaterThan(int a, int b)
        {
            return a > b;
        }

        /// <summary>
        /// Returns the sum of the two arguments: (a + b).
        /// </summary>
        /// <returns>a+b</returns>
        [ParameterNames("Sum", "A", "B")]
        public static int Plus(int a, int b)
        {
            return a + b;
        }

        /// <summary>
        /// Returns the sum of the two arguments: (a + b).
        /// </summary>
        /// <returns>a+b</returns>
        [ParameterNames("Sum", "A", "B")]
        [HasUnitDerivative]
        public static double Plus(double a, double b)
        {
            return a + b;
        }

        /// <summary>
        /// Returns the difference of the two arguments: (a - b).
        /// </summary>
        /// <returns>a-b</returns>
        [HasUnitDerivative]
        public static double Difference(double a, double b)
        {
            return a - b;
        }

        /// <summary>
        /// Returns the difference of the two arguments: (a - b).
        /// </summary>
        /// <returns>a-b</returns>
        public static int Difference(int a, int b)
        {
            return a - b;
        }

        /// <summary>
        /// Returns the product of the two arguments a * b.
        /// </summary>
        /// <returns>a*b</returns>
        public static double Product(double a, double b)
        {
            return a*b;
        }

        /// <summary>
        /// Returns the product of the two arguments a * b.
        /// </summary>
        /// <returns>a*b</returns>
        [ParameterNames("Product", "A", "B")]
        public static double Product_SHG09(double a, double b)
        {
            return a*b;
        }

        /// <summary>
        /// Returns the product  a * exp(b).
        /// </summary>
        public static double ProductExp(double a, double b)
        {
            return a*Math.Exp(b);
        }

        /// <summary>
        /// Returns the ratio of the two arguments a / b.
        /// </summary>
        /// <returns>a-b</returns>
        public static double Ratio(double a, double b)
        {
            return a/b;
        }

        /// <summary>
        /// Returns the inner product between two vectors.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        public static double InnerProduct(Vector a, Vector b)
        {
            return Vector.InnerProduct(a, b);
        }

        /// <summary>
        /// Returns the inner product between two vectors.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        [ParameterNames("X", "A", "B")]
        public static double InnerProductPartialCovariance(double[] a, Vector b)
        {
            return Vector.InnerProduct(Vector.FromArray(a), b);
        }

        /// <summary>
        /// Returns the inner product between two vectors, the first of which is binary.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns><c>sum_i a[i]*b[i]</c></returns>
        [ParameterNames("Sum", "A", "B")]
        public static double SumWhere(bool[] a, Vector b)
        {
            return Vector.InnerProduct(Vector.FromArray(a.Select(x => x ? 1.0 : 0.0).ToArray()), b);
        }

        /// <summary>
        /// Multiply matrix times vector
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns><c>sum_j a[i,j]*b[j]</c></returns>
        public static Vector Product(Matrix a, Vector b)
        {
            return a*b;
        }

        /// <summary>
        /// Multiply posdef matrix times scalar
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns><c>a[i,j]*b</c></returns>
        public static PositiveDefiniteMatrix Product(PositiveDefiniteMatrix a, double b)
        {
            return a*b;
        }

        /// <summary>
        /// Multiply scalar times posdef matrix
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns><c>a*b[i,j]</c></returns>
        public static PositiveDefiniteMatrix Product(double a, PositiveDefiniteMatrix b)
        {
            return b*a;
        }

        /// <summary>
        /// Returns the product of two matrices.
        /// </summary>
        /// <param name="A">A two-dimensional array indexed by [row,col].</param>
        /// <param name="B">A two-dimensional array indexed by [row,col].</param>
        /// <returns>A two-dimensional array indexed by [row,col].</returns>
        public static double[,] MatrixMultiply(double[,] A, double[,] B)
        {
            Assert.IsTrue(A.GetLength(1) == B.GetLength(0));
            double[,] result = new double[A.GetLength(0),B.GetLength(1)];
            for (int k = 0; k < result.GetLength(0); k++)
            {
                for (int k2 = 0; k2 < result.GetLength(1); k2++)
                {
                    double sum = 0;
                    for (int i = 0; i < A.GetLength(1); i++)
                    {
                        sum += A[k, i]*B[i, k2];
                    }
                    result[k, k2] = sum;
                }
            }
            return result;
        }

        /// <summary>
        /// Sum the numbers in an array.
        /// </summary>
        /// <param name="array"></param>
        /// <returns><c>sum_i array[i]</c></returns>
        public static double Sum(IList<double> array)
        {
            double sum = 0;
            for (int i = 0; i < array.Count; i++)
            {
                sum = sum + array[i];
            }
            return sum;
        }

        /// <summary>
        /// Sums the specified list of vectors.
        /// </summary>
        /// <param name="array">The list of vectors.</param>
        /// <returns>The sum over all vectors in the array.</returns>
        public static Vector Sum(IList<Vector> array)
        {
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            if (array.Count < 1)
            {
                throw new ArgumentException("The array must contain at least one vector.", "array");
            }

            if (array.Any(element => element == null))
            {
                throw new ArgumentNullException("array", "The array must not contain null vectors.");
            }

            int dimension = array[0].Count;
            if (array.Any(element => element.Count != dimension))
            {
                throw new ArgumentException("All vectors in the array must have the same dimensions.", "array");
            }

            Vector sum = Vector.Copy(array[0]);
            for (int i = 1; i < array.Count; i++)
            {
                sum.SetToSum(sum, array[i]);
            }

            return sum;
        }

        /// <summary>
        /// Count the number of true values in the given boolean array.
        /// </summary>
        /// <param name="array">The array of booleans.</param>
        /// <returns>Number of true values in the given array.</returns>
        [ParameterNames("count", "array")]
        public static int CountTrue(bool[] array)
        {
            return array.Count(t => t);
        }

        /// <summary>
        /// True if all array elements are true.
        /// </summary>
        /// <param name="array"></param>
        /// <returns><c>AND_i array[i]</c></returns>
        public static bool AllTrue(IList<bool> array)
        {
            bool allTrue = true;
            for (int i = 0; i < array.Count; i++)
            {
                allTrue &= array[i];
            }
            return allTrue;
        }

        /// <summary>
        /// Get an element of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="index">The index to get</param>
        /// <returns>The item</returns>
        //[ParameterNames("item", "array", "index")]
        //public static T GetItem<T>(T[] array, int index) { return array[index]; }
        [Hidden]
        [ParameterNames("item", "array", "index")]
        public static T GetItem<T>(IList<T> array, int index)
        {
            return array[index];
        }

        /// <summary>
        /// Get multiple elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items we want to get</param>
        /// <returns>The items</returns>
        [ParameterNames("items", "array", "indices")]
        public static T[] GetItems<T>(IList<T> array, IList<int> indices)
        {
            T[] result = new T[indices.Count];
            for (int i = 0; i < indices.Count; i++)
            {
                result[i] = array[indices[i]];
            }
            return result;
        }

        /// <summary>
        /// Get multiple different elements of an array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="indices">Array of indices for items we want to get.  Must all be different.</param>
        /// <returns>The items</returns>
        [ParameterNames("items", "array", "indices")]
        public static T[] Subarray<T>(IList<T> array, IList<int> indices)
        {
            return GetItems(array, indices);
        }

        [ParameterNames("items", "array", "indices")]
        public static T[][] JaggedSubarray<T>(IList<T> array, int[][] indices)
        {
            T[][] result = new T[indices.Length][];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = GetItems(array, indices[i]);
            }
            return result;
        }

        /// <summary>
        /// Get an element of a 2D array.
        /// </summary>
        /// <typeparam name="T">Type of element in the array</typeparam>
        /// <param name="array">The array</param>
        /// <param name="index1">The first index</param>
        /// <param name="index2">The second index</param>
        /// <returns>The item</returns>
        [Hidden]
        [ParameterNames("item", "array", "index1", "index2")]
        public static T GetItem2D<T>(T[,] array, int index1, int index2)
        {
            return array[index1, index2];
        }

        /// <summary>
        /// Convert a Vector into an array of doubles.
        /// </summary>
        /// <param name="vector"></param>
        /// <returns>A new array</returns>
        [ParameterNames("array", "vector")]
        public static double[] ArrayFromVector(Vector vector)
        {
            return vector.ToArray();
        }

        /// <summary>
        /// An internal factor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="def"></param>
        /// <param name="marginal"></param>
        /// <returns></returns>
        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T Variable<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "init", "marginal")]
        public static T VariableInit<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T VariableGibbs<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T VariableMax<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Hidden]
        [Stochastic]
        [ParameterNames("use", "def", "marginal")]
        public static T VariablePoint<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// An internal factor.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="def"></param>
        /// <param name="marginal"></param>
        /// <returns></returns>
        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariable<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "init", "marginal")]
        [Hidden]
        public static T DerivedVariableInit<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariableVmp<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "init", "marginal")]
        [Hidden]
        public static T DerivedVariableInitVmp<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariableMax<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "marginal")]
        [Hidden]
        public static T DerivedVariableGibbs<T>(T def, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("use", "def", "init", "marginal")]
        [Hidden]
        public static T DerivedVariableInitGibbs<T>(T def, T init, out T marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Create an array filled with a single value.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="Def">The value to fill with.</param>
        /// <param name="count"></param>
        /// <param name="Marginal">Dummy argument for inferring marginals.</param>
        /// <returns>A new array with all entries set to value.</returns>
        [Stochastic]
        [ParameterNames("Uses", "Def", "count", "Marginal")]
        [Hidden]
        public static T[] UsesEqualDef<T>(T Def, int count, out T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [Stochastic]
        [ParameterNames("Uses", "Def", "count", "burnIn", "thin", "marginal", "samples", "conditionals")]
        [Hidden]
        public static T[] UsesEqualDefGibbs<T>(T Def, int count, int burnIn, int thin, out T marginal, out T samples, out T conditionals)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Create an array filled with a single value. For reference types,
        /// the replicates all reference the same instance
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="value">The value to fill with.</param>
        /// <param name="count">Number of replicates</param>
        /// <returns>A new array with all entries set to value.</returns>
        [Hidden]
        [ParameterNames("Uses", "Def", "Count")]
        [ReturnsCompositeArray]
        public static T[] Replicate<T>([IsReturnedInEveryElement] T value, [Constant] int count)
        {
            T[] result = new T[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        /// Create a multidimensional array filled with a single value.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="value">The value to fill with.</param>
        /// <returns>A new array with all entries set to value.</returns>
        [ParameterNames("Uses", "Def")]
        public static Array ReplicateNd<T>(T value)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Create an array filled with a single value.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="Def">The value to fill with.</param>
        /// <param name="count"></param>
        /// <param name="Marginal">Dummy argument for inferring marginals.</param>
        /// <returns>A new array with all entries set to value.</returns>
        [ParameterNames("Uses", "Def", "count", "Marginal")]
        [Hidden]
        public static T[] ReplicateWithMarginal<T>(T Def, int count, out T Marginal)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        [ParameterNames("Uses", "Def", "count", "burnIn", "thin", "marginal", "samples", "conditionals")]
        [Hidden]
        public static T[] ReplicateWithMarginalGibbs<T>(T Def, int count, int burnIn, int thin, out T marginal, out T samples, out T conditionals)
        {
            throw new InvalidOperationException("Should never be called with deterministic arguments");
        }

        /// <summary>
        /// Passes the input through to the output.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="value">The value to return.</param>
        /// <returns>The supplied value.</returns>
        [Fresh] // needed for CopyPropagationTransform
        [HasUnitDerivative]
        public static T Copy<T>([SkipIfUniform] T value)
        {
            return value;
        }

        /// <summary>
        /// Passes the input through to the output. Used to set backward messages to uniform.
        /// </summary>
        /// <typeparam name="T">The type the input value.</typeparam>
        /// <param name="value">The value to return.</param>
        /// <returns>The supplied value.</returns>
        public static T Cut<T>([SkipIfUniform] T value)
        {
            return value;
        }

        /// <summary>
        /// Generate a jagged array of Gaussian random variables.
        /// </summary>
        /// <param name="sampleCount"></param>
        /// <param name="precisions"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        [Stochastic]
        [ParameterNames("samples", "sampleCount", "precisions", "indices")]
        public static double[][] GaussianArrayArray(int sampleCount, double[] precisions, out int[][] indices)
        {
            indices = Util.ArrayInit(sampleCount, i => Util.ArrayInit(precisions.Length, j => j));
            return Util.ArrayInit(sampleCount, i => Util.ArrayInit(precisions.Length, j => Factor.Gaussian(0, precisions[j])));
        }

        /// <summary>
        /// Function evaluate factor
        /// </summary>
        /// <param name="func">Function</param>
        /// <param name="x">Function input</param>
        /// <returns>Function output</returns>
        [ParameterNames("y", "func", "x")]
        public static double FunctionEvaluate(IFunction func, Vector x)
        {
            return func.Evaluate(x);
        }

        /// <summary>
        /// Rotate a 2D vector about the origin
        /// </summary>
        /// <param name="x">First coordinate of vector</param>
        /// <param name="y">Second coordinate of vector</param>
        /// <param name="angle">Counter-clockwise rotation angle in radians</param>
        /// <returns>The rotated vector</returns>
        public static Vector Rotate(double x, double y, double angle)
        {
            Vector rot = Vector.Zero(2);
            double c = Math.Cos(angle);
            double s = Math.Sin(angle);
            rot[0] = c*x - s*y;
            rot[1] = s*x + c*y;
            return rot;
        }

        /// <summary>
        /// Samples a character from a given probability distribution over characters.
        /// </summary>
        /// <param name="probabilities">A vector storing probabilities of individual characters.</param>
        /// <returns>The sampled character.</returns>
        [ParameterNames("character", "probabilities")]
        [Stochastic]
        public static char Char(Vector probabilities)
        {
            return DiscreteChar.FromVector(probabilities).Sample();
        }

        /// <summary>
        /// Converts a given character array to a string.
        /// </summary>
        /// <param name="characters">The character array to convert.</param>
        /// <returns>A string made of characters in the array.</returns>
        [ParameterNames("str", "characters")]
        public static string StringFromArray(char[] characters)
        {
            return new string(characters);
        }

        /// <summary>
        /// Samples a string from <see cref="StringDistribution.Capitalized"/> with a specified minimum string length.
        /// </summary>
        /// <param name="minLength">The minimum length of a string.</param>
        /// <returns>The sampled string.</returns>
        /// <remarks>The method always throws <see cref="InvalidOperationException"/> since the distribution is improper.</remarks>
        [ParameterNames("str", "minLength")]
        [Stochastic]
        public static string StringCapitalized(int minLength)
        {
            throw new InvalidOperationException("The distribution is improper.");
        }

        /// <summary>
        /// Samples a string from <see cref="StringDistribution.Capitalized"/> with a specified minimum and maximum string length.
        /// </summary>
        /// <param name="minLength">The minimum length of a string.</param>
        /// <param name="maxLength">The maximum length of a string.</param>
        /// <returns>The sampled string.</returns>
        [ParameterNames("str", "minLength", "maxLength")]
        [Stochastic]
        public static string StringCapitalized(int minLength, int maxLength)
        {
            return StringCapitalizedOfMinMaxLengthOp.StrAverageConditional(minLength, maxLength).Sample();
        }

        /// <summary>
        /// Samples a string from a uniform distribution over strings of a given length,
        /// with characters restricted to the support of a given distribution.
        /// </summary>
        /// <param name="length">The length of a string.</param>
        /// <param name="allowedChars">The distribution specifying allowed string characters.</param>
        /// <returns>The sampled string.</returns>
        [ParameterNames("str", "length", "allowedChars")]
        [Stochastic]
        public static string StringOfLength(int length, DiscreteChar allowedChars)
        {
            return StringOfLengthOp.StrAverageConditional(allowedChars, length).Sample();
        }

        /// <summary>
        /// Samples a string from a uniform distribution over strings of length greater or equal to a given one,
        /// with characters restricted to the support of a given distribution.
        /// </summary>
        /// <param name="minLength">The minimum length of a string.</param>
        /// <param name="allowedChars">The distribution specifying allowed string characters.</param>
        /// <returns>The sampled string.</returns>
        /// <remarks>The method always throws <see cref="InvalidOperationException"/> since the distribution is improper.</remarks>
        [ParameterNames("str", "minLength", "allowedChars")]
        [Stochastic]
        public static string String(int minLength, DiscreteChar allowedChars)
        {
            throw new InvalidOperationException("The distribution is improper.");
        }

        /// <summary>
        /// Samples a string from a uniform distribution over strings of length within given bounds,
        /// with characters restricted to the support of a given distribution.
        /// </summary>
        /// <param name="minLength">The minimum length of a string.</param>
        /// <param name="maxLength">The maximum length of a string.</param>
        /// <param name="allowedChars">The distribution specifying allowed string characters.</param>
        /// <returns>The sampled string.</returns>
        [ParameterNames("str", "minLength", "maxLength", "allowedChars")]
        [Stochastic]
        public static string String(int minLength, int maxLength, DiscreteChar allowedChars)
        {
            return StringOfMinMaxLengthOp.StrAverageConditional(allowedChars, minLength, maxLength).Sample();
        }
        
        /// <summary>
        /// Extracts a substring from a given string.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="start">The start index of the substring.</param>
        /// <param name="length">The length of the substring.</param>
        /// <returns>The extracted substring.</returns>
        [ParameterNames("sub", "str", "start", "length")]
        public static string Substring(string str, int start, int length)
        {
            return str.Substring(start, length);
        }

        /// <summary>
        /// If a given string contains a single character, extracts the character, otherwise fails.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <returns>The extracted character.</returns>
        [ParameterNames("character", "str")]
        public static char Single(string str)
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfValid(str.Length == 1, "str", "The given string must consist of a single character.");

            return str[0];
        }

        /// <summary>
        /// Concatenates given strings.
        /// </summary>
        /// <param name="str1">The first string.</param>
        /// <param name="str2">The second string.</param>
        /// <returns>The concatenation result.</returns>
        [ParameterNames("concat", "str1", "str2")]
        public static string Concat(string str1, string str2)
        {
            return str1 + str2;
        }

        /// <summary>
        /// Concatenates a character and a string.
        /// </summary>
        /// <param name="ch">The character.</param>
        /// <param name="str">The string.</param>
        /// <returns>The concatenation result.</returns>
        [ParameterNames("concat", "ch", "str")]
        public static string Concat(char ch, string str)
        {
            return ch + str;
        }

        /// <summary>
        /// Concatenates a string and a character.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="ch">The character.</param>
        /// <returns>The concatenation result.</returns>
        [ParameterNames("concat", "str", "ch")]
        public static string Concat(string str, char ch)
        {
            return str + ch;
        }

        /// <summary>
        /// Replaces argument placeholders such as {0}, {1} etc with arguments having the corresponding index,
        /// similar to what <see cref="string.Format(string, object[])"/> does.
        /// </summary>
        /// <param name="format">The string with argument placeholders.</param>
        /// <param name="args">The array of arguments.</param>
        /// <returns><paramref name="format"/> with argument placeholders replaced by arguments.</returns>
        /// <remarks>
        /// This method has the following notable differences from <see cref="string.Format(string, object[])"/>:
        /// <list type="bullet"></list>
        /// <item><description>Placeholder for each of the arguments must be present in the format string exactly once.</description></item>
        /// <item><description>No braces are allowed except for those used to specify placeholders.</description></item>
        /// </remarks>
        [ParameterNames("str", "format", "args")]
        public static string StringFormat(string format, string[] args)
        {
            StringDistribution result = StringFormatOp.StrAverageConditional(
                StringDistribution.String(format), Util.ArrayInit(args.Length, i => StringDistribution.String(args[i])));
            if (result.IsZero())
            {
                throw new ConstraintViolatedException(string.Format(
                    "The format string \"{0}\" was ill-formatted or not consistent with the argument list.",
                    format));
            }

            Debug.Assert(result.IsPointMass, "Must be a point mass.");
            return result.Point;
        }

        /// <summary>
        /// Generate a real number uniformly between [-upperBound,upperBound]
        /// </summary>
        /// <param name="upperBound"></param>
        /// <returns></returns>
        [ParameterNames("sample", "upperBound")]
        public static double UniformPlusMinus(double upperBound)
        {
            return Rand.Double()*2*upperBound - upperBound;
        }

        /// <summary>
        /// Convert an integer to a double-precision value
        /// </summary>
        /// <param name="integer"></param>
        /// <returns></returns>
        public static double Double(int integer)
        {
            return (double)integer;
        }
    }
#pragma warning disable 1591
    /// <exclude/>
    public delegate TRet FactorMethod<TRet>();

    /// <exclude/>
    public delegate TRet FactorMethod<TRet, T1>(T1 arg1);

    /// <exclude/>
    public delegate TRet FactorMethod<TRet, T1, T2>(T1 arg1, T2 arg2);

    /// <exclude/>
    public delegate TRet FactorMethod<TRet, T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);

    /// <exclude/>
    public delegate TRet FactorMethod<TRet, T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);
#pragma warning restore 1591

    /// <summary>
    /// Delegate definition for constraint method with one argument
    /// </summary>
    /// <typeparam name="T1">Argument type for the constraint method</typeparam>
    /// <param name="arg1">Argument for the constraint method</param>
    /// <exclude/>
    public delegate void ConstrainMethod<T1>(T1 arg1);

    /// <summary>
    /// Delegate definition for constraint method with two arguments
    /// </summary>
    /// <typeparam name="T1">First argument type for the constraint method</typeparam>
    /// <typeparam name="T2">Second argument type for the constraint method</typeparam>
    /// <param name="arg1">First argument for the constraint method</param>
    /// <param name="arg2">Second argument for the constraint method</param>
    /// <exclude/>
    public delegate void ConstrainMethod<T1, T2>(T1 arg1, T2 arg2);

    /// <summary>
    /// Delegate definition for constraint method with three arguments
    /// </summary>
    /// <typeparam name="T1">First argument type for the constraint method</typeparam>
    /// <typeparam name="T2">Second argument type for the constraint method</typeparam>
    /// <typeparam name="T3">Third argument type for the constraint method</typeparam>
    /// <param name="arg1">First argument for the constraint method</param>
    /// <param name="arg2">Second argument for the constraint method</param>
    /// <param name="arg3">Third argument for the constraint method</param>
    /// <exclude/>
    public delegate void ConstrainMethod<T1, T2, T3>(T1 arg1, T2 arg2, T3 arg3);

    /// <summary>
    /// Delegate definition for constraint method with four arguments
    /// </summary>
    /// <typeparam name="T1">First argument type for the constraint method</typeparam>
    /// <typeparam name="T2">Second argument type for the constraint method</typeparam>
    /// <typeparam name="T3">Third argument type for the constraint method</typeparam>
    /// <typeparam name="T4">Fourth argument type for the constraint method</typeparam>
    /// <param name="arg1">First argument for the constraint method</param>
    /// <param name="arg2">Second argument for the constraint method</param>
    /// <param name="arg3">Third argument for the constraint method</param>
    /// <param name="arg4">Fourth argument for the constraint method</param>
    /// <exclude/>
    public delegate void ConstrainMethod<T1, T2, T3, T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);

    public delegate TResult FuncOut<T1, TOut, TResult>(T1 arg, out TOut output);

    public delegate TResult FuncOut<T1, T2, TOut, TResult>(T1 arg, T2 arg2, out TOut output);

    public delegate TResult FuncOut2<T1, TOut, TOut2, TResult>(T1 arg, out TOut output, out TOut2 output2);

    public delegate TResult FuncOut2<T1, T2, TOut, TOut2, TResult>(T1 arg, T2 arg2, out TOut output, out TOut2 output2);

    public delegate TResult FuncOut3<T1, T2, TOut, TOut2, TOut3, TResult>(T1 arg, T2 arg2, out TOut output, out TOut2 output2, out TOut3 output3);

    public delegate TResult FuncOut3<T1, T2, T3, TOut, TOut2, TOut3, TResult>(T1 arg, T2 arg2, T3 arg3, out TOut output, out TOut2 output2, out TOut3 output3);

    public delegate TResult FuncOut3<T1, T2, T3, T4, TOut, TOut2, TOut3, TResult>(T1 arg, T2 arg2, T3 arg3, T4 arg4, out TOut output, out TOut2 output2, out TOut3 output3);

    public delegate TResult FuncOut4<T1, T2, TOut, TOut2, TOut3, TOut4, TResult>(T1 arg, T2 arg2, out TOut output, out TOut2 output2, out TOut3 output3, out TOut4 output4);

    public delegate TResult FuncOut4<T1, T2, T3, TOut, TOut2, TOut3, TOut4, TResult>(
        T1 arg, T2 arg2, T3 arg3, out TOut output, out TOut2 output2, out TOut3 output3, out TOut4 output4);
}
