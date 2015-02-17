/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Factors;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// A base class for implementations of distributions over sequences.
    /// </summary>
    /// <typeparam name="TSequence">The type of a sequence.</typeparam>
    /// <typeparam name="TElement">The type of a sequence element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
    /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
    /// <typeparam name="TWeightFunction">The type of an underlying function mapping sequences to weights. Currently must be a weighted finite state automaton.</typeparam>
    /// <typeparam name="TThis">The type of a concrete distribution class.</typeparam>
    [Serializable]
    [Quality(QualityBand.Experimental)]
    public abstract class SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis> :
        IDistribution<TSequence>,
        SettableTo<TThis>,
        SettableToProduct<TThis>,
        SettableToRatio<TThis>,
        SettableToPower<TThis>,
        CanGetLogAverageOf<TThis>,
        CanGetLogAverageOfPower<TThis>,
        CanGetAverageLog<TThis>,
        SettableToWeightedSumExact<TThis>,
        SettableToPartialUniform<TThis>,
        CanGetLogNormalizer,
        Sampleable<TSequence>
        where TSequence : class, IEnumerable<TElement>
        where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
        where TElementDistribution : class, IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
        where TWeightFunction : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>, new()
        where TThis : SequenceDistribution<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis>, new()
    {
        #region Fields & constants

        /// <summary>
        /// A sequence manipulator.
        /// </summary>
        private static readonly TSequenceManipulator SequenceManipulator = new TSequenceManipulator();

        /// <summary>
        /// A function mapping sequences to weights (non-normalized probabilities).
        /// </summary>
        private TWeightFunction sequenceToWeight;

        /// <summary>
        /// Specifies whether the <see cref="sequenceToWeight"/> is normalized.
        /// </summary>
        private bool isNormalized;

        /// <summary>
        /// If the distribution is a point mass, stores the point. Otherwise it is set to <see langword="null"/>.
        /// </summary>
        private TSequence point;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}"/> class
        /// by setting the underlying weight function to be zero everywhere.
        /// </summary>
        protected SequenceDistribution()
        {
            // TODO: should it be uniform by default to comply to the rest of Infer.NET?
            this.sequenceToWeight = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Zero();
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets the point mass represented by the distribution.
        /// </summary>
        public TSequence Point
        {
            get
            {
                if (this.point == null)
                {
                    throw new InvalidOperationException("This distribution is not a point mass.");
                }

                return this.point;
            }

            set
            {
                Argument.CheckIfNotNull(value, "value", "Point mass must not be null.");
                
                this.sequenceToWeight = null;
                this.point = value;
                this.isNormalized = true;
            }
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution represents a point mass.
        /// </summary>
        public bool IsPointMass
        {
            get { return this.point != null; }
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution
        /// puts all probability mass on the empty sequence.
        /// </summary>
        public bool IsEmpty
        {
            get { return this.IsPointMass && SequenceManipulator.GetLength(this.Point) == 0; }
        }

        #endregion

        #region Factory methods

        /// <summary>
        /// Creates a point mass distribution.
        /// </summary>
        /// <param name="point">The point.</param>
        /// <returns>The created point mass distribution.</returns>
        [Construction("Point", UseWhen = "IsPointMass")]
        public static TThis PointMass(TSequence point)
        {
            Argument.CheckIfNotNull(point, "point", "Point mass must not be null.");

            return new TThis { Point = point };
        }

        /// <summary>
        /// Creates an improper distribution which assigns the probability of 1 to every sequence.
        /// </summary>
        /// <returns>The created uniform distribution.</returns>
        [Construction(UseWhen = "IsUniform")]
        [Skip]
        public static TThis Uniform()
        {
            var result = new TThis();
            result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Creates an improper distribution which assigns zero probability to every sequence.
        /// </summary>
        /// <returns>The created zero distribution.</returns>
        [Construction(UseWhen = "IsZero")]
        public static TThis Zero()
        {
            var result = new TThis();
            result.SetToZero();
            return result;
        }

        /// <summary>
        /// Creates a distribution from a given weight (non-normalized probability) function.
        /// </summary>
        /// <param name="sequenceToWeight">The weight function specifying the distribution.</param>
        /// <returns>The created distribution.</returns>
        [Construction("GetWorkspace")]
        public static TThis FromWeightFunction(TWeightFunction sequenceToWeight)
        {
            Argument.CheckIfNotNull(sequenceToWeight, "sequenceToWeight");

            return FromWorkspace(sequenceToWeight.Clone());
        }

        /// <summary>
        /// Creates a distribution which will use a given weight function as a workspace.
        /// Any modifications to the workspace after the distribution has been created
        /// would put the distribution into an invalid state.
        /// </summary>
        /// <param name="workspace">The workspace to create the distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis FromWorkspace(TWeightFunction workspace)
        {
            Argument.CheckIfNotNull(workspace, "workspace");

            var result = new TThis();
            result.SetWorkspace(workspace);
            return result;
        }

        /// <summary>
        /// Creates a distribution which puts all probability mass on the empty sequence.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsEmpty")]
        public static TThis Empty()
        {
            return PointMass(SequenceManipulator.ToSequence(new TElement[0]));
        }

        /// <summary>
        /// Creates a distribution over sequences of length 1 induced by a given distribution over sequence elements.
        /// </summary>
        /// <param name="elementDistribution">The distribution over sequence elements.</param>
        /// <returns>The created distribution.</returns>
        /// <remarks>
        /// The distribution created by this method can differ from the result of
        /// <see cref="Repeat(TElementDistribution, int, int?)"/> with both min and max number of times to repeat set to 1 since the latter always creates a partial uniform distribution.
        /// </remarks>
        public static TThis SingleElement(TElementDistribution elementDistribution)
        {
            var func = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Zero();
            var end = func.Start.AddTransition(elementDistribution, 0.0);
            end.EndWeight = 1.0;
            return FromWorkspace(func);
        }

        /// <summary>
        /// Creates a distribution which puts all probability mass on a sequence containing only a given element.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <returns>The created distribution.</returns>
        public static TThis SingleElement(TElement element)
        {
            var sequence = SequenceManipulator.ToSequence(new List<TElement> { element });
            return PointMass(sequence);
        }

        /// <summary>
        /// Creates a distribution which is a uniform mixture of a given set of distributions.
        /// </summary>
        /// <param name="distributions">The set of distributions to create a mixture from.</param>
        /// <returns>The created mixture distribution.</returns>
        public static TThis OneOf(IEnumerable<TThis> distributions)
        {
            Argument.CheckIfNotNull(distributions, "distributions");

            var enumerable = distributions as IList<TThis> ?? distributions.ToList();
            if (enumerable.Count == 1)
            {
                return enumerable[0].Clone();
            }

            var probFunctions = enumerable.Select(d => d.GetProbabilityFunction());
            return FromWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Sum(probFunctions));
        }

        /// <summary>
        /// Creates a distribution which is a uniform mixture of a given set of distributions.
        /// </summary>
        /// <param name="distributions">The set of distributions to create a mixture from.</param>
        /// <returns>The created mixture distribution.</returns>
        public static TThis OneOf(params TThis[] distributions)
        {
            return OneOf((IEnumerable<TThis>)distributions);
        }

        /// <summary>
        /// Creates a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        /// <returns>The created mixture distribution.</returns>
        public static TThis OneOf(double weight1, TThis dist1, double weight2, TThis dist2)
        {
            var result = new TThis();
            result.SetToSum(weight1, dist1, weight2, dist2);
            return result;
        }

        /// <summary>
        /// Creates a distribution which assigns specified probabilities to given sequences.
        /// Probabilities do not have to be normalized.
        /// </summary>
        /// <param name="sequenceProbPairs">A list of (sequence, probability) pairs.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(IEnumerable<KeyValuePair<TSequence, double>> sequenceProbPairs)
        {
            return FromWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.FromValues(sequenceProbPairs));
        }

        /// <summary>
        /// Creates a distribution which is uniform over a given set of sequences.
        /// </summary>
        /// <param name="sequences">The set of sequences to create a distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(IEnumerable<TSequence> sequences)
        {
            Argument.CheckIfNotNull(sequences, "sequences");

            var enumerable = sequences as IList<TSequence> ?? sequences.ToList();
            if (enumerable.Count == 1)
            {
                return PointMass(enumerable[0]);
            }

            return OneOf(enumerable.Select(PointMass));
        }

        /// <summary>
        /// Creates a distribution which is uniform over a given set of sequences.
        /// </summary>
        /// <param name="sequences">The set of sequences to create a distribution from.</param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOf(params TSequence[] sequences)
        {
            return OneOf((IEnumerable<TSequence>)sequences);
        }

        /// <summary>
        /// Creates a uniform distribution over sequences of length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible sequence length.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Any(int minLength = 0, int? maxLength = null)
        {
            return Repeat(Distribution.CreateUniform<TElementDistribution>(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over sequences of length within the given bounds.
        /// Sequence elements are restricted to be equal to a given element.
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <param name="minTimes">The minimum possible sequence length. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TElement element, int minTimes = 1, int? maxTimes = null)
        {
            var elementDistribution = new TElementDistribution { Point = element };
            return Repeat(elementDistribution, minTimes, maxTimes);
        }

        /// <summary>
        /// Creates a uniform distribution over sequences of length within the given bounds.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <param name="minTimes">The minimum possible sequence length. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TElementDistribution allowedElements, int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfNotNull(allowedElements, "allowedElements");
            Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of times to repeat must be non-negative.");
            Argument.CheckIfInRange(!maxTimes.HasValue || maxTimes.Value >= 0, "maxTimes", "The maximum number of times to repeat must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || minTimes <= maxTimes.Value, "The minimum length cannot be greater than the maximum length.");

            //// TODO: delegate to Repeat(TThis, int, int?)
            
            if (maxTimes.HasValue)
            {
                if (minTimes == 0 && maxTimes.Value == 0)
                {
                    return Empty();
                }

                if (allowedElements.IsPointMass && (minTimes == maxTimes.Value))
                {
                    return PointMass(SequenceManipulator.ToSequence(Enumerable.Repeat(allowedElements.Point, minTimes)));
                }
            }

            allowedElements = Distribution.CreatePartialUniform(allowedElements);
            double distLogNormalizer = -allowedElements.GetLogAverageOf(allowedElements);
            var func = Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Zero();
            var state = func.Start;

            int iterationBound = maxTimes.HasValue ? maxTimes.Value : minTimes;
            for (int i = 0; i <= iterationBound; i++)
            {
                bool isLengthAllowed = i >= minTimes;
                state.EndLogWeight = isLengthAllowed ? 0 : double.NegativeInfinity;
                if (i < iterationBound)
                {
                    state = state.AddTransition(allowedElements, distLogNormalizer); // todo: clone set?    
                }
            }

            if (!maxTimes.HasValue)
            {
                state.AddSelfTransition(allowedElements, distLogNormalizer);
            }

            return FromWorkspace(func);
        }

        /// <summary>
        /// <para>
        /// Creates a distribution by applying <see cref="Automaton{TSequence, TElement, TElementDistribution, TSequenceManipulator, TThis}.Repeat(TThis, int, int?)"/>
        /// to the weight function of a given distribution,
        /// which is additionally scaled by the inverse of <see cref="GetLogAverageOf"/> with itself.
        /// So, if the given distribution is partial uniform, the result will be partial uniform over the repetitions of
        /// sequences covered by the distribution.
        /// </para>
        /// <para>
        /// If <paramref name="maxTimes"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </para>
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="minTimes">The minimum number of repetitions. Defaults to 1.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis Repeat(TThis dist, int minTimes = 1, int? maxTimes = null)
        {
            Argument.CheckIfNotNull(dist, "dist");
            Argument.CheckIfInRange(minTimes >= 0, "minTimes", "The minimum number of repetitions must be non-negative.");
            Argument.CheckIfValid(!maxTimes.HasValue || maxTimes.Value >= minTimes, "The maximum number of repetitions must not be less than the minimum number.");

            if (dist.IsPointMass && maxTimes.HasValue && minTimes == maxTimes)
            {
                var newSequenceElements = new List<TElement>(SequenceManipulator.GetLength(dist.Point) * minTimes);
                for (int i = 0; i < minTimes; ++i)
                {
                    newSequenceElements.AddRange(dist.Point);
                }

                return PointMass(SequenceManipulator.ToSequence(newSequenceElements));
            }

            double logNormalizer = -dist.GetLogAverageOf(dist);
            return FromWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Repeat(
                dist.GetWorkspaceOrPoint().ScaleLog(logNormalizer), minTimes, maxTimes));
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TThis, int, int?)"/> with the minimum number of repetitions set to 0.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TThis dist, int? maxTimes = null)
        {
            return Repeat(dist, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElement, int, int?)"/> with the minimum number of repetitions set to 0.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TElement element, int? maxTimes = null)
        {
            return Repeat(element, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElementDistribution, int, int?)"/> with the minimum number of repetitions set to 0.
        /// </summary>
        /// <param name="allowedElements">The allowed sequence elements.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis ZeroOrMore(TElementDistribution allowedElements, int? maxTimes = null)
        {
            return Repeat(allowedElements, minTimes: 0, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TThis, int, int?)"/> with the minimum number of repetitions set to 1.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TThis dist, int? maxTimes = null)
        {
            return Repeat(dist, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElement, int, int?)"/> with the minimum number of repetitions set to 1.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TElement element, int? maxTimes = null)
        {
            return Repeat(element, maxTimes: maxTimes);
        }

        /// <summary>
        /// An alias for <see cref="Repeat(TElementDistribution, int, int?)"/> with the minimum number of repetitions set to 0.
        /// </summary>
        /// <param name="allowedElements">The allowed sequence elements.</param>
        /// <param name="maxTimes">
        /// The maximum number of repetitions, or <see langword="null"/> for no upper bound.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static TThis OneOrMore(TElementDistribution allowedElements, int? maxTimes = null)
        {
            return Repeat(allowedElements, maxTimes: maxTimes);
        }

        /// <summary>
        /// Creates a mixture of a given distribution and a point mass representing an empty sequence.
        /// </summary>
        /// <param name="dist">The distribution.</param>
        /// <param name="prob">The probability of the component corresponding to <paramref name="dist"/>.</param>
        /// <returns>The created mixture.</returns>
        public static TThis Optional(TThis dist, double prob = 0.5)
        {
            return OneOf(prob, dist, 1 - prob, Empty());
        }

        #endregion

        #region Manipulation

        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution and a given element.
        /// </summary>
        /// <param name="element">The element to append.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append the given element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences and the element.</returns>
        public TThis Append(TElement element, byte group = 0)
        {
            return this.Append(SingleElement(element), group);
        }
        
        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution
        /// and elements from a given distribution.
        /// </summary>
        /// <param name="elementDistribution">The distribution to generate the elements from.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// 2) Sample a random element from <paramref name="elementDistribution"/>.
        /// </description></item>
        /// <item><description>
        /// 3) Append the sampled element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences and elements.</returns>
        public TThis Append(TElementDistribution elementDistribution, byte group = 0)
        {
            return this.Append(SingleElement(elementDistribution), group);
        }

        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution and a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append <paramref name="sequence"/> to it and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences.</returns>
        public TThis Append(TSequence sequence, byte group = 0)
        {
            return this.Append(PointMass(sequence), group);
        }

        /// <summary>
        /// Creates a distribution over concatenations of sequences from the current distribution
        /// and sequences from a given distribution.
        /// </summary>
        /// <param name="dist">The distribution over the sequences to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Sample a random sequence from <paramref name="dist"/>.
        /// </description></item>
        /// <item><description>
        /// Output the concatenation of the sampled pair of sequences.
        /// </description></item>
        /// </list>
        /// </remarks>
        /// <returns>The distribution over the concatenations of sequences.</returns>
        public TThis Append(TThis dist, byte group = 0)
        {
            Argument.CheckIfNotNull(dist, "dist");
            
            TThis result = this.Clone();
            result.AppendInPlace(dist, group);
            return result;
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and a given element.
        /// </summary>
        /// <param name="element">The element to append.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append the given element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TElement element, byte group = 0)
        {
            this.AppendInPlace(SingleElement(element), group);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and elements from a given distribution.
        /// </summary>
        /// <param name="elementDistribution">The distribution to generate the elements from.</param>
        /// <param name="group">The group for the appended element.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Sample a random element from <paramref name="elementDistribution"/>.
        /// </description></item>
        /// <item><description>
        /// Append the sampled element to the sampled sequence and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TElementDistribution elementDistribution, byte group = 0)
        {
            Argument.CheckIfNotNull(elementDistribution, "elementDistribution");
            
            this.AppendInPlace(SingleElement(elementDistribution), group);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and a given sequence.
        /// </summary>
        /// <param name="sequence">The sequence to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Append <paramref name="sequence"/> to it and output the result.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TSequence sequence, byte group = 0)
        {
            Argument.CheckIfNotNull(sequence, "sequence");
            
            this.AppendInPlace(PointMass(sequence), group);
        }

        /// <summary>
        /// Replaces the current distribution by a distribution over concatenations of sequences
        /// from the current distribution and sequences from a given distribution.
        /// </summary>
        /// <param name="dist">The distribution over the sequences to append.</param>
        /// <param name="group">The group for the appended sequence.</param>
        /// <remarks>
        /// The result is equivalent to the distribution produced by the following sampling procedure:
        /// <list type="number">
        /// <item><description>
        /// Sample a random sequence from the current distribution.
        /// </description></item>
        /// <item><description>
        /// Sample a random sequence from <paramref name="dist"/>.
        /// </description></item>
        /// <item><description>
        /// Output the concatenation of the sampled pair of sequences.
        /// </description></item>
        /// </list>
        /// </remarks>
        public void AppendInPlace(TThis dist, byte group = 0)
        {
            Argument.CheckIfNotNull(dist, "dist");

            if (this.IsPointMass && dist.IsPointMass && group == 0)
            {
                this.Point = SequenceManipulator.Concat(this.point, dist.point);
                return;
            }

            var workspace = this.GetWorkspaceOrPoint();
            if (dist.IsPointMass)
            {
                workspace.AppendInPlace(dist.Point, group);
            }
            else
            {
                workspace.AppendInPlace(dist.sequenceToWeight, group);
            }

            this.SetWorkspace(workspace);
        }

        /// <summary>
        /// Gets a weight function that maps sequences to their probabilities under this distribution.
        /// </summary>
        /// <returns>The function mapping sequences to probabilities.</returns>
        public TWeightFunction GetProbabilityFunction()
        {
            if (this.IsPointMass)
            {
                return Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.ConstantOn(
                    1.0, this.point);
            }
            
            this.EnsureNormalized();
            return this.sequenceToWeight.Clone();
        }

        /// <summary>
        /// Replaces the current distribution by a distribution induced by a given weight function.
        /// </summary>
        /// <param name="newSequenceToWeight">The function mapping sequences to weights.</param>
        public void SetWeightFunction(TWeightFunction newSequenceToWeight)
        {
            Argument.CheckIfNotNull(newSequenceToWeight, "newSequenceToWeight");

            this.SetWorkspace(newSequenceToWeight.Clone());
        }

        /// <summary>
        /// Returns the underlying weight function, or <see langword="null"/>, if the distribution is a point mass.
        /// Any modifications of the returned function will put the distribution into an undefined state.
        /// </summary>
        /// <returns>The underlying weight function, or <see langword="null"/> if the distribution is a point mass.</returns>
        public TWeightFunction GetWorkspace()
        {
            this.EnsureNormalized();
            return this.sequenceToWeight;
        }

        /// <summary>
        /// Returns the underlying weight function, or, if the distribution is a point mass,
        /// a functional representation of the corresponding point.
        /// Any modifications of the returned function will put the distribution into an undefined state.
        /// </summary>
        /// <returns>The underlying weight function or a functional representation of the point.</returns>
        public TWeightFunction GetWorkspaceOrPoint()
        {
            return this.IsPointMass
                ? Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.ConstantOn(1.0, this.point)
                : this.GetWorkspace();
        }

        #endregion

        #region ToString

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <returns>
        /// A string that represents the distribution.
        /// </returns>
        public override string ToString()
        {
            return this.ToString(SequenceDistributionFormats.Friendly);
        }

        /// <summary>
        /// Returns a string that represents the distribution.
        /// </summary>
        /// <param name="format">The format.</param>
        /// <returns>A string that represents the distribution.</returns>
        public string ToString(ISequenceDistributionFormat format)
        {
            Argument.CheckIfNotNull(format, "format");

            return format.ConvertToString<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis>((TThis)this);
        }

        #endregion

        #region Groups

        public void PruneToGroup(byte groupToKeep)
        {
            if (this.IsPointMass)
            {
                return; // TODO: get rid of groups or do something about groups + point mass combo
            }

            this.sequenceToWeight.PruneToGroup(groupToKeep);
            this.NormalizeStructure();
        }

        #endregion

        #region IDistribution implementation

        /// <summary>
        /// Replaces the current distribution by a copy of the given distribution.
        /// </summary>
        /// <param name="that">The distribution to set the current distribution to.</param>
        public void SetTo(TThis that)
        {
            Argument.CheckIfNotNull(that, "that");
            
            if (ReferenceEquals(this, that))
            {
                return;
            }

            this.point = that.point;
            if (this.point == null)
            {
                this.SetWeightFunction(that.sequenceToWeight);
            }
            else
            {
                this.sequenceToWeight = null;
            }

            this.isNormalized = that.isNormalized;
        }

        /// <summary>
        /// Replaces the current distribution by an improper zero distribution.
        /// </summary>
        public void SetToZero()
        {
            this.SetWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.Zero());
            this.isNormalized = true;
        }

        /// <summary>
        /// Returns a product of the current distribution and a given one.
        /// </summary>
        /// <param name="that">The distribution to compute the product with.</param>
        /// <returns>The product.</returns>
        public TThis Product(TThis that)
        {
            Argument.CheckIfNotNull(that, "that");
            
            var auto = new TThis();
            auto.SetToProduct((TThis)this, that);
            return auto;
        }

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions.
        /// </summary>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        public void SetToProduct(TThis dist1, TThis dist2)
        {
            Argument.CheckIfNotNull(dist1, "dist1");
            Argument.CheckIfNotNull(dist2, "dist2");
            
            this.DoSetToProduct(dist1, dist2, computeLogNormalizer: false);
        }

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions
        /// Returns the logarithm of the normalizer for the product (as returned by <see cref="GetLogAverageOf"/>).
        /// </summary>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        /// <returns>The logarithm of the normalizer for the product.</returns>
        public double SetToProductAndReturnLogNormalizer(TThis dist1, TThis dist2)
        {
            Argument.CheckIfNotNull(dist1, "dist1");
            Argument.CheckIfNotNull(dist2, "dist2");
            
            double? logNormalizer = this.DoSetToProduct(dist1, dist2, computeLogNormalizer: true);
            Debug.Assert(logNormalizer.HasValue, "Log-normalizer wasn't computed.");
            return logNormalizer.Value;
        }

        /// <summary>
        /// Returns the logarithm of the probability that the current distribution would draw the same sample
        /// as a given one.
        /// </summary>
        /// <param name="that">The given distribution.</param>
        /// <returns>The logarithm of the probability that distributions would draw the same sample.</returns>
        public double GetLogAverageOf(TThis that)
        {
            Argument.CheckIfNotNull(that, "that");
            
            if (that.IsPointMass)
            {
                return this.GetLogProb(that.point);
            }

            if (this.IsPointMass)
            {
                return that.GetLogProb(this.point);
            }

            var temp = new TThis();
            return temp.SetToProductAndReturnLogNormalizer((TThis)this, that);
        }

        /// <summary>
        /// Computes the log-integral of one distribution times another raised to a power.
        /// </summary>
        /// <param name="that">The other distribution</param>
        /// <param name="power">The exponent</param>
        /// <returns><c>Math.Log(sum_x this.Evaluate(x) * Math.Pow(that.Evaluate(x), power))</c></returns>
        /// <remarks>
        /// <para>
        /// This is not the same as GetLogAverageOf(that^power) because it includes the normalization constant of that.
        /// </para>
        /// <para>Powers other than 1 are not currently supported.</para>
        /// </remarks>
        public double GetLogAverageOfPower(TThis that, double power)
        {
            if (power == 1.0)
            {
                return this.GetLogAverageOf(that);
            }
            
            throw new NotImplementedException("GetLogAverageOfPower() is not implemented for non-unit power.");
        }

        /// <summary>
        /// Computes the expected logarithm of a given distribution under this distribution.
        /// Not currently supported.
        /// </summary>
        /// <param name="that">The distribution to take the logarithm of.</param>
        /// <returns><c>sum_x this.Evaluate(x)*Math.Log(that.Evaluate(x))</c></returns>
        /// <remarks>This is also known as the cross entropy.</remarks>
        public double GetAverageLog(TThis that)
        {
            throw new NotSupportedException("GetAverageLog() is not supported for sequence distributions.");
        }

        /// <summary>
        /// Returns the logarithm of the normalizer of the exponential family representation of this distribution.
        /// Normalizer of an improper distribution is defined to be 1.
        /// </summary>
        /// <returns>The logarithm of the normalizer.</returns>
        /// <remarks>Getting the normalizer is currently supported for improper distributions only.</remarks>
        public double GetLogNormalizer()
        {
            if (!this.IsProper())
            {
                return 0;
            }

            throw new NotSupportedException("GetLogNormalizer() is not supported for proper distributions.");
        }

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="weight1">The weight of the first distribution.</param>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="weight2">The weight of the second distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        public void SetToSum(double weight1, TThis dist1, double weight2, TThis dist2)
        {
            Argument.CheckIfValid(weight1 + weight2 >= 0, "The sum of the weights must not be negative.");
            
            if (weight1 < 0 || weight2 < 0)
            {
                throw new NotImplementedException("Negative weights are not yet supported.");
            }

            this.SetToSumLog(Math.Log(weight1), dist1, Math.Log(weight2), dist2);
        }

        /// <summary>
        /// Replaces the current distribution with a mixture of a given pair of distributions.
        /// </summary>
        /// <param name="logWeight1">The logarithm of the weight of the first distribution.</param>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="logWeight2">The logarithm of the weight of the second distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        public void SetToSumLog(double logWeight1, TThis dist1, double logWeight2, TThis dist2)
        {
            Argument.CheckIfNotNull(dist1, "dist1");
            Argument.CheckIfNotNull(dist2, "dist2");
            Argument.CheckIfValid(!double.IsPositiveInfinity(logWeight1) || !double.IsPositiveInfinity(logWeight2), "Both weights are infinite.");
            
            if (double.IsNegativeInfinity(logWeight1) && double.IsNegativeInfinity(logWeight2))
            {
                this.SetToUniform();
                return;
            }

            if (double.IsPositiveInfinity(logWeight1) || double.IsNegativeInfinity(logWeight2))
            {
                this.SetTo(dist1);
                return;
            }

            if (double.IsPositiveInfinity(logWeight2) || double.IsNegativeInfinity(logWeight1))
            {
                this.SetTo(dist2);
                return;
            }

            double weightSum = MMath.LogSumExp(logWeight1, logWeight2);
            logWeight1 -= weightSum;
            logWeight2 -= weightSum;

            dist1.EnsureNormalized();
            dist2.EnsureNormalized();

            this.SetWorkspace(
                Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.WeightedSumLog(
                    logWeight1, dist1.GetWorkspaceOrPoint(), logWeight2, dist2.GetWorkspaceOrPoint()));
        }

        /// <summary>
        /// Replaces the current distribution with a given distribution raised to a given power.
        /// </summary>
        /// <param name="that">The distribution to raise to the power.</param>
        /// <param name="power">The power.</param>
        /// <remarks>Only 0 and 1 are currently supported as powers.</remarks>
        public void SetToPower(TThis that, double power)
        {
            if (power == 0.0)
            {
                this.SetToUniform();
                return;
            }
            
            if (power != 1.0)
            {
                throw new NotSupportedException("SetToPower() is not supported for powers other than 0 and 1.");
            }
        }

        /// <summary>
        /// Replaces the current distribution with the ratio of a given pair of distributions. Not currently supported.
        /// </summary>
        /// <param name="numerator">The numerator in the ratio.</param>
        /// <param name="denominator">The denominator in the ratio.</param>
        /// <param name="forceProper">Specifies whether the ratio must be proper.</param>
        public void SetToRatio(TThis numerator, TThis denominator, bool forceProper)
        {
            throw new NotSupportedException("SetToRatio() is not supported for sequence distributions.");
        }

        /// <summary>
        /// Creates a copy of the current distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        object ICloneable.Clone()
        {
            return this.Clone();
        }

        /// <summary>
        /// Creates a copy of the current distribution.
        /// </summary>
        /// <returns>The created copy.</returns>
        public TThis Clone()
        {
            var result = new TThis();
            result.SetTo((TThis)this);
            return result;
        }

        /// <summary>
        /// Gets a value indicating how close this distribution is to a given one
        /// in terms of probabilities they assign to sequences.
        /// </summary>
        /// <param name="that">The other distribution.</param>
        /// <returns>A non-negative value, which is close to zero if the two distribution assign similar values to all sequences.</returns>
        public double MaxDiff(object that)
        {
            TThis thatDistribution = that as TThis;
            if (thatDistribution == null)
            {
                return double.PositiveInfinity;
            }
            
            this.EnsureNormalized();
            thatDistribution.EnsureNormalized();
            return Math.Exp(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.GetLogSimilarity(
                this.GetWorkspaceOrPoint(), thatDistribution.GetWorkspaceOrPoint()));
        }

        /// <summary>
        /// Gets the logarithm of the probability of a given sequence under this distribution.
        /// If the distribution is improper, returns the logarithm of the value of the underlying unnormalized weight function.
        /// </summary>
        /// <param name="sequence">The sequence to get the probability for.</param>
        /// <returns>The logarithm of the probability of the sequence.</returns>
        public double GetLogProb(TSequence sequence)
        {
            Argument.CheckIfNotNull(sequence, "sequence");
            
            if (this.IsPointMass)
            {
                if (SequenceManipulator.SequencesAreEqual(sequence, this.point))
                {
                    return 0;
                }

                return double.NegativeInfinity;
            }

            this.EnsureNormalized();
            return this.sequenceToWeight.GetLogValue(sequence);
        }

        /// <summary>
        /// Gets the probability of a given sequence under this distribution.
        /// If the distribution is improper, returns the value of the underlying unnormalized weight function.
        /// </summary>
        /// <param name="sequence">The sequence to get the probability for.</param>
        /// <returns>The probability of the sequence.</returns>
        public double Evaluate(TSequence sequence)
        {
            return Math.Exp(this.GetLogProb(sequence));
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <param name="result">A pre-allocated storage for the sample (will be ignored).</param>
        /// <returns>The drawn sample.</returns>
        public TSequence Sample(TSequence result)
        {
            return this.Sample();
        }

        /// <summary>
        /// Draws a sample from the distribution.
        /// </summary>
        /// <returns>The drawn sample.</returns>
        public TSequence Sample()
        {
            if (this.IsPointMass)
            {
                return this.Point;
            }
            
            if (!this.IsProper())
            {
                throw new InvalidOperationException("Can't sample from an improper distribution!");
            }

            this.EnsureNormalized();
            
            var sampledElements = new List<TElement>();
            var currentState = this.sequenceToWeight.Start;
            while (true)
            {
                double logSample = Math.Log(Rand.Double());
                double logProbSum = double.NegativeInfinity;
                for (int i = 0; i < currentState.Transitions.Count; ++i)
                {
                    var transition = currentState.Transitions[i];
                
                    logProbSum = MMath.LogSumExp(logProbSum, transition.LogWeight);
                    if (logSample < logProbSum)
                    {
                        if (!transition.IsEpsilon)
                        {
                            sampledElements.Add(transition.ElementDistribution.Sample());
                        }

                        currentState = this.sequenceToWeight.States[transition.DestinationStateIndex];
                        break;
                    }
                }    

                if (logSample >= logProbSum)
                {
                    Debug.Assert(!double.IsNegativeInfinity(currentState.EndLogWeight), "This state must have a non-zero ending probability.");
                    return SequenceManipulator.ToSequence(sampledElements);
                }
            }
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns the probability of 1 to every sequence.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        public void SetToUniformOf(TElementDistribution allowedElements)
        {
            this.SetToUniformOf(allowedElements, 0.0);
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns the probability of 1 to every sequence.
        /// </summary>
        public void SetToUniform()
        {
            this.SetToUniformOf(Distribution.CreateUniform<TElementDistribution>());
        }

        /// <summary>
        /// Sets the distribution to be uniform over its support.
        /// </summary>
        public void SetToPartialUniform()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Sets the distribution to be uniform over the support of a given distribution.
        /// </summary>
        /// <param name="dist">The distribution which support will be used to setup the current distribution.</param>
        public void SetToPartialUniformOf(TThis dist)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Checks whether the distribution is uniform over its support.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution is uniform over its support, <see langword="false"/> otherwise.</returns>
        public bool IsPartialUniform()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Checks whether the current distribution is proper.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is proper, <see langword="false"/> otherwise.</returns>
        public bool IsProper()
        {
            if (this.IsPointMass)
            {
                return true;
            }

            return !double.IsInfinity(this.sequenceToWeight.GetLogNormalizer()); // TODO: cache?
        }
        
        /// <summary>
        /// Checks whether the current distribution is uniform over all possible sequences.
        /// </summary>
        /// <returns><see langword="true"/> if the current distribution is uniform over all possible sequences, <see langword="false"/> otherwise.</returns>
        public bool IsUniform()
        {
            if (this.IsPointMass)
            {
                return false;
            }

            TThis canonicUniform = Uniform();
            return this.Equals(canonicUniform);
        }

        /// <summary>
        /// Gets a value indicating whether the current distribution
        /// is an improper distribution which assigns zero probability to every sequence.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> if the current distribution
        /// is an improper distribution that assigns zero probability to every sequence, <see langword="false"/> otherwise.
        /// </returns>
        public bool IsZero()
        {
            return !this.IsPointMass && this.sequenceToWeight.IsZero();
        }

        /// <summary>
        /// Checks if <paramref name="obj"/> equals to this distribution (i.e. represents the same distribution over sequences).
        /// </summary>
        /// <param name="obj">The object to compare this distribution with.</param>
        /// <returns><see langword="true"/> if this distribution is equal to <paramref name="obj"/>, false otherwise.</returns>
        public override bool Equals(object obj)
        {
            if (obj == null || obj.GetType() != typeof(TThis))
            {
                return false;
            }

            TThis that = (TThis)obj;
            this.EnsureNormalized();
            that.EnsureNormalized();
            return this.GetWorkspaceOrPoint().Equals(that.GetWorkspaceOrPoint());
        }

        /// <summary>
        /// Gets the hash code of this distribution.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            return this.GetWorkspaceOrPoint().GetHashCode();
        }

        #endregion

        #region Helpers

        #region Helpers for producing improper distributions

        /// <summary>
        /// Creates an improper distribution which assigns a given probability to every sequence.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <param name="uniformLogProb">The logarithm of the probability assigned to every allowed sequence.</param>
        /// <returns>The created distribution.</returns>
        protected static TThis UniformOf(TElementDistribution allowedElements, double uniformLogProb)
        {
            var result = new TThis();
            result.SetToUniformOf(allowedElements, uniformLogProb);
            return result;
        }

        /// <summary>
        /// Replaces the current distribution with an improper distribution which assigns a given probability to every sequence.
        /// Sequence elements are restricted to be non-zero probability elements from a given distribution.
        /// </summary>
        /// <param name="allowedElements">The distribution representing allowed sequence elements.</param>
        /// <param name="uniformLogProb">The logarithm of the probability assigned to every allowed sequence.</param>
        protected void SetToUniformOf(TElementDistribution allowedElements, double uniformLogProb)
        {
            Argument.CheckIfNotNull(allowedElements, "allowedElements");

            this.SetWorkspace(Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction>.ConstantLog(uniformLogProb, allowedElements));
        }

        #endregion

        /// <summary>
        /// Replaces the current distribution with a product of a given pair of distributions,
        /// optionally normalizing the result.
        /// </summary>
        /// <param name="dist1">The first distribution.</param>
        /// <param name="dist2">The second distribution.</param>
        /// <param name="computeLogNormalizer">Specifies whether the product normalizer should be computed and used to normalize the product.</param>
        /// <returns>The logarithm of the normalizer for the product if requested, <see langword="null"/> otherwise.</returns>
        private double? DoSetToProduct(TThis dist1, TThis dist2, bool computeLogNormalizer)
        {
            Debug.Assert(dist1 != null && dist2 != null, "Valid distributions must be provided.");
            
            if (dist1.IsPointMass)
            {
                var pt = dist1.Point;
                double lognorm = dist2.GetLogProb(pt);
                if (!double.IsNegativeInfinity(lognorm))
                {
                    this.SetTo(dist1);
                }
                else
                {
                    this.SetToZero();
                }

                return computeLogNormalizer ? (double?)lognorm : null;
            }

            if (dist2.IsPointMass)
            {
                var pt = dist2.Point;
                double lognorm = dist1.GetLogProb(pt);
                if (double.IsNegativeInfinity(lognorm))
                {
                    this.SetToZero();
                    return computeLogNormalizer ? (double?)lognorm : null;
                }

                if (!dist1.UsesGroups())
                {
                    this.Point = pt;
                    return computeLogNormalizer ? (double?)lognorm : null;
                }
            }

            if (computeLogNormalizer)
            {
                dist1.EnsureNormalized();
                dist2.EnsureNormalized();    
            }
            
            var weightFunction1 = dist1.GetWorkspaceOrPoint();
            var weightFunction2 = dist2.GetWorkspaceOrPoint();
            var product = weightFunction1.Product(weightFunction2);
            this.SetWorkspace(product);

            double? logNormalizer = null;
            if (computeLogNormalizer)
            {
                if (product.IsZero())
                {
                    logNormalizer = double.NegativeInfinity; // Normalizer of zero is defined to be -inf, not zero
                }
                else
                {
                    double computedLogNormalizer;
                    if (!product.TryNormalizeValues(out computedLogNormalizer))
                    {
                        computedLogNormalizer = 0;
                    }

                    logNormalizer = computedLogNormalizer;    
                }
                
                this.isNormalized = true;
            }
            
            return logNormalizer;
        }
        
        /// <summary>
        /// Checks if the distribution uses groups.
        /// </summary>
        /// <returns><see langword="true"/> if the distribution uses groups, <see langword="false"/> otherwise.</returns>
        private bool UsesGroups()
        {
            if (this.IsPointMass)
            {
                return false;
            }

            return this.sequenceToWeight.UsesGroups();
        }

        /// <summary>
        /// Replaces the workspace (weight function) of the current distribution with a given one.
        /// </summary>
        /// <param name="workspace">The workspace to replace the current one with.</param>
        /// <remarks>
        /// If the given workspace represents a point mass, the distribution would be converted to a point mass
        /// and the workspace would be set to <see langword="null"/>.
        /// </remarks>
        private void SetWorkspace(TWeightFunction workspace)
        {
            Debug.Assert(workspace != null, "The new workspace cannot be null.");

            this.point = null;
            this.sequenceToWeight = workspace;
            this.isNormalized = false;

            this.NormalizeStructure();
        }

        /// <summary>
        /// Normalizes the underlying weight function (if there is one), if it hasn't been normalized before.
        /// If the distribution is improper, does nothing but marks it as normalized.
        /// </summary>
        private void EnsureNormalized()
        {
            if (!this.isNormalized)
            {
                this.sequenceToWeight.TryNormalizeValues();
                this.isNormalized = true;
            }
        }
        
        /// <summary>
        /// Modifies the distribution to be in normalized form e.g. using special
        /// case structures for point masses.
        /// </summary>
        private void NormalizeStructure()
        {
            if (this.UsesGroups())
            {
                return; // todo: remove groups
            }

            var pt = this.TryComputePoint();
            if (pt != null)
            {
                this.Point = pt;
            }
        }

        /// <summary>
        /// Returns a point mass represented by the current distribution, or <see langword="null"/>,
        /// if it doesn't represent a point mass.
        /// </summary>
        /// <returns>The point mass represented by the distribution, or <see langword="null"/>.</returns>
        private TSequence TryComputePoint()
        {
            if (this.IsPointMass)
            {
                return this.point;
            }

            return this.sequenceToWeight.TryComputePoint();
        }
        
        #endregion
    }
}
