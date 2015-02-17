/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Distributions
{
    using System;

    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// Represents a distribution over strings that uses a weighted finite state automaton as the underlying weight function.
    /// </summary>
    [Serializable]
    [Quality(QualityBand.Experimental)]
    public class StringDistribution :
        SequenceDistribution<string, char, DiscreteChar, StringManipulator, StringAutomaton, StringDistribution>
    {
        /// <summary>
        /// Concatenates the weighted regular languages defined by given distributions
        /// (see <see cref="SequenceDistribution{TSequence,TElement,TElementDistribution,TSequenceManipulator,TWeightFunction,TThis}.Append(TThis, byte)"/>).
        /// </summary>
        /// <param name="first">The first distribution.</param>
        /// <param name="second">The second distribution.</param>
        /// <returns>The concatenation result.</returns>
        public static StringDistribution operator +(StringDistribution first, StringDistribution second)
        {
            return first.Append(second);
        }

        /// <summary>
        /// Creates a point mass distribution.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis}.PointMass"/>.
        /// </summary>
        /// <param name="str">The point.</param>
        /// <returns>The created point mass distribution.</returns>
        public static StringDistribution String(string str)
        {
            return StringDistribution.PointMass(str);
        }
        
        /// <summary>
        /// Creates a distribution which puts all mass on a string containing only a given character.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis}.SingleElement(TElement)"/>.
        /// </summary>
        /// <param name="ch">The character.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Char(char ch)
        {
            return StringDistribution.SingleElement(ch);
        }

        /// <summary>
        /// Creates a distribution over strings of length 1 induced by a given distribution over characters.
        /// This method is an alias for <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis}.SingleElement(TElementDistribution)"/>.
        /// </summary>
        /// <param name="characterDist">The distribution over characters.</param>
        /// <returns>The created distribution.</returns>
        /// <remarks>
        /// The distribution created by this method can differ from the result of
        /// <see cref="SequenceDistribution{TSequence, TElement, TElementDistribution, TSequenceManipulator, TWeightFunction, TThis}.Repeat(TThis, int, int?)"/>
        /// with both min and max length set to 1 since the latter always creates a partial uniform distribution.
        /// </remarks>
        public static StringDistribution Char(DiscreteChar characterDist)
        {
            return StringDistribution.SingleElement(characterDist);
        }
        
        /// <summary>
        /// Creates a uniform distribution over all strings that are case-invariant matches of the specified string.
        /// </summary>
        /// <param name="template">The string to match.</param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution CaseInvariant(string template)
        {
            StringDistribution result = StringDistribution.Empty();
            foreach (var ch in template)
            {
                var upper = char.ToUpperInvariant(ch);
                var lower = char.ToLowerInvariant(ch);
                if (upper == lower)
                {
                    result.AppendInPlace(ch);
                }
                else
                {
                    result.AppendInPlace(DiscreteChar.OneOf(lower, upper));
                }
            }
            
            return result;
        }

        /// <summary>
        /// Creates a uniform distribution over strings of lowercase letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Lower(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.Lower(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of uppercase letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Upper(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.Upper(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of lowercase and uppercase letters, with length in given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Letters(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.Letter(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of digits, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Digits(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.Digit(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of digits, lowercase and uppercase letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution LettersOrDigits(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.LetterOrDigit(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of word characters (see <see cref="DiscreteChar.WordChar"/>),
        /// with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution WordChars(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.WordChar(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings of whitespace characters (see <see cref="DiscreteChar.Whitespace"/>),
        /// with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 1.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Whitespace(int minLength = 1, int? maxLength = null)
        {
            return StringDistribution.Repeat(DiscreteChar.Whitespace(), minLength, maxLength);
        }

        /// <summary>
        /// Creates a uniform distribution over strings that start with an upper case letter followed by
        /// one or more lower case letters, with length within the given bounds.
        /// If <paramref name="maxLength"/> is set to <see langword="null"/>,
        /// there will be no upper bound on the length, and the resulting distribution will thus be improper.
        /// </summary>
        /// <param name="minLength">The minimum possible string length. Defaults to 2.</param>
        /// <param name="maxLength">
        /// The maximum possible sequence length, or <see langword="null"/> for no upper bound on length.
        /// Defaults to <see langword="null"/>.
        /// </param>
        /// <returns>The created distribution.</returns>
        public static StringDistribution Capitalized(int minLength = 2, int? maxLength = null)
        {
            Argument.CheckIfInRange(minLength >= 2, "minLength", "The minimum length of a capitalized string should be 2 or more.");
            Argument.CheckIfValid(!maxLength.HasValue || maxLength.Value >= minLength, "The maximum length cannot be less than the minimum length.");

            var result = StringDistribution.Char(DiscreteChar.Upper());
            
            if (maxLength.HasValue)
            {
                result.AppendInPlace(StringDistribution.Lower(minLength: minLength - 1, maxLength: maxLength.Value - 1));
            }
            else
            {
                // Concatenation with an improper distribution, need to adjust its scale so that the result is 1 on its support
                double logNormalizer = result.GetLogAverageOf(result);
                var lowercaseSuffixFunc = StringDistribution.Lower(minLength: minLength - 1).GetProbabilityFunction();
                var lowercaseSuffixFuncScaled = lowercaseSuffixFunc.ScaleLog(-logNormalizer);
                result.AppendInPlace(StringDistribution.FromWorkspace(lowercaseSuffixFuncScaled));    
            }

            return result;
        }
    }
}
