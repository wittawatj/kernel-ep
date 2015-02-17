// (C) Copyright 2008 Microsoft Research Cambridge

namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for <see cref="Factor.Product(Matrix, Vector)" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(Matrix), typeof(Vector))]
    [Buffers("BMean", "BVariance")]
    [Quality(QualityBand.Stable)]
    public static class MatrixVectorProductOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector product, Matrix A, Vector B)
        {
            return product.Equals(Factor.Product(A, B)) ? 0.0 : Double.NegativeInfinity;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector product, Matrix A, Vector B)
        {
            return LogAverageFactor(product, A, B);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Vector product, Matrix A, Vector B)
        {
            return LogAverageFactor(product, A, B);
        }

        /// <summary>Initialize the buffer <c>BVariance</c>.</summary>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Initial value of buffer <c>BVariance</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static PositiveDefiniteMatrix BVarianceInit([IgnoreDependency] VectorGaussian B)
        {
            return new PositiveDefiniteMatrix(B.Dimension, B.Dimension);
        }

        /// <summary>Update the buffer <c>BVariance</c>.</summary>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static PositiveDefiniteMatrix BVariance([Proper] VectorGaussian B, PositiveDefiniteMatrix result)
        {
            return B.GetVariance(result);
        }

        /// <summary>Initialize the buffer <c>BMean</c>.</summary>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Initial value of buffer <c>BMean</c>.</returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static Vector BMeanInit([IgnoreDependency] VectorGaussian B)
        {
            return Vector.Zero(B.Dimension);
        }

        /// <summary>Update the buffer <c>BMean</c>.</summary>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="BVariance">Buffer <c>BVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para />
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Vector BMean([Proper] VectorGaussian B, [Fresh] PositiveDefiniteMatrix BVariance, Vector result)
        {
            return B.GetMean(result, BVariance);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="to_product">Outgoing message to <c>product</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product) p(product) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(VectorGaussian product, [Fresh] VectorGaussian to_product)
        {
            return to_product.GetLogAverageOf(product);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="BMean">Buffer <c>BMean</c>.</param>
        /// <param name="BVariance">Buffer <c>BVariance</c>.</param>
        /// <returns>Logarithm of the factor's average value across the given argument distributions.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(product,a,b))</c>.</para>
        /// </remarks>
        public static double LogAverageFactor(Vector product, Matrix A, VectorGaussian B, [Fresh] Vector BMean, [Fresh] PositiveDefiniteMatrix BVariance)
        {
            VectorGaussian toProduct = ProductAverageConditional(A, BMean, BVariance, new VectorGaussian(A.Rows));
            return toProduct.GetLogProb(product);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,b) p(product,b) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(VectorGaussian product, Matrix A, VectorGaussian B)
        {
            return 0.0;
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(VectorGaussian product, Matrix A, VectorGaussian B)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="BMean">Buffer <c>BMean</c>.</param>
        /// <param name="BVariance">Buffer <c>BVariance</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(b) p(b) factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(Vector product, Matrix A, VectorGaussian B, [Fresh] Vector BMean, [Fresh] PositiveDefiniteMatrix BVariance)
        {
            return LogAverageFactor(product, A, B, BMean, BVariance);
        }

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>.</param>
        /// <param name="BMean">Buffer <c>BMean</c>.</param>
        /// <param name="BVariance">Buffer <c>BVariance</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        public static double AverageLogFactor(Vector product, Matrix A, VectorGaussian B, [Fresh] Vector BMean, [Fresh] PositiveDefiniteMatrix BVariance)
        {
            return LogAverageFactor(product, A, B, BMean, BVariance);
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="BMean">Buffer <c>BMean</c>.</param>
        /// <param name="BVariance">Buffer <c>BVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>product</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian ProductAverageConditional(
            Matrix A, [Fresh] Vector BMean, [Fresh] PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            // P.mean = A*B.mean
            // P.var = A*B.var*A'
            // if A is invertible, then
            // P.prec = inv(A)'*inv(B.var)*inv(A)
            // P.precTimesMean = inv(A)'*B.precTimesMean
            Vector rmean = A * BMean;
            PositiveDefiniteMatrix rvariance = new PositiveDefiniteMatrix(result.Dimension, result.Dimension);
            Matrix temp = (A * BVariance).Transpose();
            rvariance.SetToProduct(A, temp);
            result.SetMeanAndVariance(rmean, rvariance);
            return result;
        }

        /// <summary />
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian ProductAverageConditionalInit([IgnoreDependency] Matrix A)
        {
            return new VectorGaussian(A.Rows);
        }

        /// <summary />
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns />
        /// <remarks>
        ///   <para />
        /// </remarks>
        [Skip]
        public static VectorGaussian ProductAverageLogarithmInit([IgnoreDependency] Matrix A)
        {
            return new VectorGaussian(A.Rows);
        }

        /// <summary>VMP message to <c>product</c>.</summary>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="BMean">Buffer <c>BMean</c>.</param>
        /// <param name="BVariance">Buffer <c>BVariance</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>product</c> conditioned on the given values.</para>
        /// </remarks>
        public static VectorGaussian ProductAverageLogarithm(
            Matrix A, [Fresh] Vector BMean, [Fresh] PositiveDefiniteMatrix BVariance, VectorGaussian result)
        {
            return ProductAverageConditional(A, BMean, BVariance, result);
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        public static VectorGaussian BAverageConditional(VectorGaussian product, Matrix A, VectorGaussian result)
        {
            if (product.IsPointMass)
                return BAverageConditional(product.Point, A, result);
            //   (p.mean - A*B)'*p.prec*(p.mean - A*B)
            // = B'*(A'*p.prec*A)*B - B'*A'*p.prec*p.mean - ...
            // B.prec = A'*p.prec*A
            // B.precTimesMean = A'*p.precTimesMean
            if (true)
            {
                // this method is slower but more numerically accurate
                // L*L' = p.prec
                int dim = product.Precision.Cols;
                LowerTriangularMatrix L = new LowerTriangularMatrix(dim, dim);
                L.SetToCholesky(product.Precision);
                Matrix At = A.Transpose();
                Matrix temp = At * L;
                result.Precision.SetToOuter(temp);
            }
            else
            {
                Matrix temp = (product.Precision * A).Transpose();
                result.Precision.SetToProduct(temp, A);
            }
            result.MeanTimesPrecision.SetToProduct(product.MeanTimesPrecision, A);
            return result;
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>product</c> integrated out. The formula is <c>sum_product p(product) factor(product,a,b)</c>.</para>
        /// </remarks>
        public static VectorGaussian BAverageLogarithm(VectorGaussian product, Matrix A, VectorGaussian result)
        {
            return BAverageConditional(product, A, result);
        }

        private const string LowRankNotSupportedMessage = "A matrix-vector product with fixed output is not yet implemented.";

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        [NotSupported(MatrixVectorProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian BAverageConditional(Vector product, Matrix A, VectorGaussian result)
        {
            throw new NotSupportedException(MatrixVectorProductOp.LowRankNotSupportedMessage);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        [NotSupported(MatrixVectorProductOp.LowRankNotSupportedMessage)]
        public static VectorGaussian BAverageLogarithm(Vector product, Matrix A, VectorGaussian result)
        {
            throw new NotSupportedException(MatrixVectorProductOp.LowRankNotSupportedMessage);
        }
    }
}
