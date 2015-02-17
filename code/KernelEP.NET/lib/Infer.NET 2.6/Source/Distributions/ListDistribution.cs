/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Distributions
{
    using System;
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// A base class for distributions over lists that use a weighted finite state automaton as the underlying weight function.
    /// </summary>
    /// <typeparam name="TList">The type of a list.</typeparam>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over list elements.</typeparam>
    /// <typeparam name="TThis">The type of a concrete distribution class.</typeparam>
    [Serializable]
    [Quality(QualityBand.Experimental)]
    public abstract class ListDistribution<TList, TElement, TElementDistribution, TThis> :
        SequenceDistribution<TList, TElement, TElementDistribution, ListManipulator<TList, TElement>, ListAutomaton<TList, TElement, TElementDistribution>, TThis>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : class, IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
        where TThis : ListDistribution<TList, TElement, TElementDistribution, TThis>, new()
    {
    }

    /// <summary>
    /// Represents a distribution over lists that use a weighted finite state automaton as the underlying weight function.
    /// </summary>
    /// <typeparam name="TList">The type of a list.</typeparam>
    /// <typeparam name="TElement">The type of a list element.</typeparam>
    /// <typeparam name="TElementDistribution">The type of a distribution over list elements.</typeparam>
    [Serializable]
    [Quality(QualityBand.Experimental)]
    public class ListDistribution<TList, TElement, TElementDistribution> :
        ListDistribution<TList, TElement, TElementDistribution, ListDistribution<TList, TElement, TElementDistribution>>
        where TList : class, IList<TElement>, new()
        where TElementDistribution : class, IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>, SettableToPartialUniform<TElementDistribution>, Sampleable<TElement>, new()
    {
    }
}
