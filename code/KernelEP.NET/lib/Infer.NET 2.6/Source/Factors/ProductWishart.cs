namespace MicrosoftResearch.Infer.Factors
{
    using System;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>Provides outgoing messages for the following factors:<list type="bullet"><item><description><see cref="Factor.Product(PositiveDefiniteMatrix, double)" /></description></item><item><description><see cref="Factor.Product(double, PositiveDefiniteMatrix)" /></description></item></list>, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "Product", typeof(PositiveDefiniteMatrix), typeof(double))]
    [FactorMethod(new[] { "Product", "b", "a" }, typeof(Factor), "Product", typeof(double), typeof(PositiveDefiniteMatrix))]
    [Quality(QualityBand.Experimental)]
    public static class ProductWishartOp
    {
        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(product,a) p(product,a) factor(product,a,b) / sum_product p(product) messageTo(product))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(Wishart product, Wishart a, double b)
        {
            return 0.0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="product">Constant value for <c>product</c>.</param>
        /// <param name="a">Incoming message from <c>a</c>.</param>
        /// <param name="b">Constant value for <c>b</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(a) p(a) factor(product,a,b))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix product, Wishart a, double b)
        {
            throw new NotImplementedException();
        }

        /// <summary>EP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[p(product) sum_(a) p(a) factor(product,a,b)]/p(product)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Wishart ProductAverageConditional([SkipIfUniform] Wishart A, double B, Wishart result)
        {
            if (A.IsPointMass)
            {
                result.Rate.SetTo(A.Point);
                result.Rate.Scale(B);
            }
            else
            {
                result.SetTo(A);
                result.Rate.Scale(1/B);
            }
            return result;
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>b</c> as the random arguments are varied. The formula is <c>proj[p(b) sum_(product) p(product) factor(product,a,b)]/p(b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gamma BAverageConditional([SkipIfUniform] Wishart Product, PositiveDefiniteMatrix A)
        {
            if (Product.IsPointMass) return BAverageLogarithm(Product.Point, A);
            // (ab)^(shape-(d+1)/2) exp(-tr(rate*(ab)))
            return Gamma.FromShapeAndRate(Product.Shape + (1 - (Product.Dimension + 1)*0.5), Matrix.TraceOfProduct(Product.Rate, A));
        }

        /// <summary>EP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing EP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma BAverageConditional(PositiveDefiniteMatrix Product, PositiveDefiniteMatrix A)
        {
            if (Product.Count == 0) return Gamma.Uniform();
            bool allZeroA = true;
            double ratio = 0;
            for (int i = 0; i < Product.Count; i++)
            {
                if (A[i] != 0)
                {
                    ratio = Product[i]/A[i];
                    allZeroA = false;
                }
            }
            if (allZeroA) return Gamma.Uniform();
            for (int i = 0; i < Product.Count; i++)
            {
                if (Math.Abs(Product[i] - A[i]*ratio) > 1e-15) throw new ConstraintViolatedException("Product is not a multiple of B");
            }
            return Gamma.PointMass(ratio);
        }

        /// <summary>EP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>a</c> as the random arguments are varied. The formula is <c>proj[p(a) sum_(product) p(product) factor(product,a,b)]/p(a)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Wishart AAverageConditional([SkipIfUniform] Wishart Product, double B, Wishart result)
        {
            result.SetTo(Product);
            result.Rate.Scale(B);
            return result;
        }

        //- VMP ----------------------------------------------------------------------------------------------------------------------

        /// <summary>Evidence message for VMP.</summary>
        /// <param name="product">Incoming message from <c>product</c>.</param>
        /// <returns>Zero.</returns>
        /// <remarks>
        ///   <para>In Variational Message Passing, the evidence contribution of a deterministic factor is zero. Adding up these values across all factors and variables gives the log-evidence estimate for VMP.</para>
        /// </remarks>
        [Skip]
        public static double AverageLogFactor(Wishart product)
        {
            return 0.0;
        }

        /// <summary>VMP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[sum_(a,b) p(a,b) factor(product,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Wishart ProductAverageLogarithm([SkipIfUniform] Wishart A, [SkipIfUniform] Gamma B, Wishart result)
        {
            if (B.IsPointMass) return ProductAverageLogarithm(A, B.Point, result);
            // E[x] = E[a]*E[b]
            // E[log(x)] = E[log(a)]+E[log(b)]
            PositiveDefiniteMatrix m = new PositiveDefiniteMatrix(A.Dimension, A.Dimension);
            A.GetMean(m);
            m.Scale(B.GetMean());
            double meanLogDet = A.Dimension*B.GetMeanLog() + A.GetMeanLogDeterminant();
            if (m.LogDeterminant() < meanLogDet) throw new MatrixSingularException(m);
            return Wishart.FromMeanAndMeanLogDeterminant(m, meanLogDet, result);
        }

        /// <summary>VMP message to <c>product</c>.</summary>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>product</c> as the random arguments are varied. The formula is <c>proj[sum_(a) p(a) factor(product,a,b)]</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Wishart ProductAverageLogarithm([SkipIfUniform] Wishart A, double B, Wishart result)
        {
            return ProductAverageConditional(A, B, result);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. Because the factor is deterministic, <c>product</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(a) p(a) log(sum_product p(product) factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        public static Gamma BAverageLogarithm([SkipIfUniform] Wishart Product, [Proper] Wishart A)
        {
            if (A.IsPointMass) return BAverageLogarithm(Product, A.Point);
            if (Product.IsPointMass) return BAverageLogarithm(Product.Point, A);
            // (ab)^(shape-(d+1)/2) exp(-tr(rate*(ab)))
            return Gamma.FromShapeAndRate(Product.Shape + (1 - (Product.Dimension + 1)*0.5), Matrix.TraceOfProduct(Product.Rate, A.GetMean()));
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> with <c>product</c> integrated out. The formula is <c>sum_product p(product) factor(product,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Gamma BAverageLogarithm([SkipIfUniform] Wishart Product, [Proper] PositiveDefiniteMatrix A)
        {
            return BAverageConditional(Product, A);
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="A">Constant value for <c>a</c>.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>b</c> conditioned on the given values.</para>
        /// </remarks>
        public static Gamma BAverageLogarithm(PositiveDefiniteMatrix Product, PositiveDefiniteMatrix A)
        {
            return BAverageConditional(Product, A);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. Because the factor is deterministic, <c>product</c> is integrated out before taking the logarithm. The formula is <c>exp(sum_(b) p(b) log(sum_product p(product) factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        public static Wishart AAverageLogarithm([SkipIfUniform] Wishart Product, [Proper] Gamma B, Wishart result)
        {
            if (B.IsPointMass) return AAverageLogarithm(Product, B.Point, result);
            if (Product.IsPointMass) return AAverageLogarithm(Product.Point, B, result);
            // (ab)^(shape-1) exp(-rate*(ab))
            result.Shape = Product.Shape;
            result.Rate.SetToProduct(Product.Rate, B.GetMean());
            return result;
        }

        /// <summary>VMP message to <c>b</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="A">Incoming message from <c>a</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <returns>The outgoing VMP message to the <c>b</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>b</c>. The formula is <c>exp(sum_(a) p(a) log(factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="A" /> is not a proper distribution.</exception>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Gamma BAverageLogarithm(PositiveDefiniteMatrix Product, [Proper] Wishart A)
        {
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="B">Incoming message from <c>b</c>. Must be a proper distribution. If uniform, the result will be uniform.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the exponential of the average log-factor value, where the average is over all arguments except <c>a</c>. The formula is <c>exp(sum_(b) p(b) log(factor(product,a,b)))</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="B" /> is not a proper distribution.</exception>
        [NotSupported(GaussianProductVmpOp.NotSupportedMessage)]
        public static Wishart AAverageLogarithm(PositiveDefiniteMatrix Product, [Proper] Gamma B, Wishart result)
        {
            throw new NotSupportedException(GaussianProductVmpOp.NotSupportedMessage);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Incoming message from <c>product</c>. Must be a proper distribution. If any element is uniform, the result will be uniform.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> with <c>product</c> integrated out. The formula is <c>sum_product p(product) factor(product,a,b)</c>.</para>
        /// </remarks>
        /// <exception cref="ImproperMessageException">
        ///   <paramref name="Product" /> is not a proper distribution.</exception>
        public static Wishart AAverageLogarithm([SkipIfUniform] Wishart Product, double B, Wishart result)
        {
            if (Product.IsPointMass) return AAverageLogarithm(Product.Point, B, result);
            return AAverageConditional(Product, B, result);
        }

        /// <summary>VMP message to <c>a</c>.</summary>
        /// <param name="Product">Constant value for <c>product</c>.</param>
        /// <param name="B">Constant value for <c>b</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is the factor viewed as a function of <c>a</c> conditioned on the given values.</para>
        /// </remarks>
        public static Wishart AAverageLogarithm(PositiveDefiniteMatrix Product, double B, Wishart result)
        {
            result.Point = Product;
            result.Point.Scale(1 / B);
            return result;
        }
    }

#if false
	[FactorMethod(typeof(Factor), "Product", typeof(PositiveDefiniteMatrix), typeof(double))]
	[FactorMethod(new string[] { "Product", "b", "a" }, typeof(Factor), "Product", typeof(double), typeof(PositiveDefiniteMatrix))]
	[Buffers("Q")]
	[Quality(QualityBand.Experimental)]
	public static class ProductWishartOp_Laplace
	{
		public static Wishart ProductAverageConditional(Wishart Product, Wishart A, Gamma B, Wishart result)
		{
		}
	}
#endif
}
