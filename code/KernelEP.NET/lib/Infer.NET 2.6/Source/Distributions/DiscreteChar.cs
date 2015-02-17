/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

using System.Globalization;

namespace MicrosoftResearch.Infer.Distributions
{
    using System.Collections.Generic;

    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>
    /// A discrete distribution over characters.
    /// </summary>
    [Quality(QualityBand.Preview)]
    public class DiscreteChar : GenericDiscreteBase<char, DiscreteChar>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DiscreteChar"/> class with a uniform distribution.
        /// </summary>
        public DiscreteChar() :
            base(1 + (int)char.MaxValue, Sparsity.Piecewise)
        {
        }

        #region Properties

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Digit"/>.
        /// </summary>
        public bool IsDigit
        {
            get { return this.Equals(Digit()); }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Lower"/>.
        /// </summary>
        public bool IsLower
        {
            get { return this.Equals(Lower()); }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Upper"/>.
        /// </summary>
        public bool IsUpper
        {
            get { return this.Equals(Upper()); }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="Letter"/>.
        /// </summary>
        public bool IsLetter
        {
            get { return this.Equals(Letter()); }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="LetterOrDigit"/>.
        /// </summary>
        public bool IsLetterOrDigit
        {
            get { return this.Equals(LetterOrDigit()); }
        }

        /// <summary>
        /// Gets a value indicating whether this distribution equals the distribution created by <see cref="WordChar"/>.
        /// </summary>
        public bool IsWordChar
        {
            get { return this.Equals(LetterOrDigit()); }
        }

        #endregion

        #region Factories

        /// <summary>
        /// Creates a uniform distribution over digits '0'..'9'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsDigit")]
        public static DiscreteChar Digit()
        {
            return DiscreteChar.InRange('0', '9');
        }

        /// <summary>
        /// Creates a uniform distribution over lowercase letters 'a'..'z'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsLower")]
        public static DiscreteChar Lower()
        {
            return DiscreteChar.InRange('a', 'z');
        }

        /// <summary>
        /// Creates a uniform distribution over uppercase letters 'A'..'Z'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        [Construction(UseWhen = "IsUpper")]
        public static DiscreteChar Upper()
        {
            return DiscreteChar.InRange('A', 'Z');
        }

        /// <summary>
        /// Creates a uniform distribution over letters in 'a'..'z' and 'A'..'Z'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar Letter()
        {
            return DiscreteChar.InRanges("azAZ");
        }

        /// <summary>
        /// Creates a uniform distribution over 'a'..'z', 'A'..'Z' and '0'..'9'.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar LetterOrDigit()
        {
            return DiscreteChar.InRanges("azAZ09");
        }

        /// <summary>
        /// Creates a uniform distribution over word characters ('a'..'z', 'A'..'Z', '0'..'9', '_' and '\'').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar WordChar()
        {
            return DiscreteChar.InRanges("azAZ09__''");
        }

        /// <summary>
        /// Creates a uniform distribution over all characters except ('a'..'z', 'A'..'Z', '0'..'9', '_' and '\'').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar NonWordChar()
        {
            return DiscreteChar.WordChar().Complement();
        }

        /// <summary>
        /// Creates a uniform distribution over whitespace characters ('\t'..'\r', ' ').
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar Whitespace()
        {
            return DiscreteChar.InRanges("\t\r  ");
        }

        /// <summary>
        /// Creates a uniform distribution over all characters.
        /// This method is an alias for <see cref="GenericDiscreteBase{T, TThis}.Uniform"/>.
        /// </summary>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar Any()
        {
            return DiscreteChar.Uniform();
        }

        /// <summary>
        /// Creates a uniform distribution over characters in a given range.
        /// This method is an alias for <see cref="GenericDiscreteBase{T, TThis}.UniformInRange"/>.
        /// </summary>
        /// <param name="start">The start of the range (inclusive).</param>
        /// <param name="end">The end of the range (inclusive).</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar InRange(char start, char end)
        {
            return DiscreteChar.UniformInRange(start, end);
        }

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in an array whose length must therefore be even.
        /// This method is an alias for <see cref="GenericDiscreteBase{T, TThis}.UniformInRanges(T[])"/>.
        /// </summary>
        /// <param name="startEndPairs">The array of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar InRanges(params char[] startEndPairs)
        {
            return DiscreteChar.UniformInRanges(startEndPairs);
        }

        /// <summary>
        /// Creates a distribution which is uniform over values in
        /// multiple ranges specified by pairs of start and end values. These pairs
        /// are specified as adjacent values in a sequence whose length must therefore be even.
        /// This method is an alias for <see cref="GenericDiscreteBase{T, TThis}.UniformInRanges(IEnumerable{T})"/>.
        /// </summary>
        /// <param name="startEndPairs">The sequence of range starts and ends.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar InRanges(IEnumerable<char> startEndPairs)
        {
            return DiscreteChar.UniformInRanges(startEndPairs);
        }

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// This method is an alias for <see cref="GenericDiscreteBase{T, TThis}.UniformOver(T[])"/>.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar OneOf(params char[] chars)
        {
            return DiscreteChar.UniformOver(chars);
        }

        /// <summary>
        /// Creates a distribution which is uniform over the specified set of characters.
        /// This method is an alias for <see cref="GenericDiscreteBase{T, TThis}.UniformOver(IEnumerable{T})"/>.
        /// </summary>
        /// <param name="chars">The characters.</param>
        /// <returns>The created distribution.</returns>
        public static DiscreteChar OneOf(IEnumerable<char> chars)
        {
            return DiscreteChar.UniformOver(chars);
        }

        #endregion

        #region Operations

        /// <summary>
        /// Creates a distribution which is uniform over all characters
        /// that have zero probability under this distribution
        /// i.e. that are not 'in' this distribution.
        /// </summary>
        /// <remarks>
        /// This is useful for defining characters that are not in a particular distribution
        /// e.g. not a letter or not a word character.
        /// </remarks>
        /// <returns>The created distribution.</returns>
        public DiscreteChar Complement()
        {
            // This creates a vector whose common value is not zero,
            // but where the piece values are zero.  This is useful when
            // displaying the distribution (to show that it is a 'complement')
            // but may have unforeseen side effects e.g. on performance.
            // todo: consider revisiting this design.
            PiecewiseVector res;
            if (this.IsPointMass)
            {
                res = PiecewiseVector.Constant(this.Dimension, 1.0);
                res[this.Point] = 0;
            }
            else
            {
                res = PiecewiseVector.Zero(this.Dimension);
                res.SetToFunction(this.disc.GetWorkspace(), x => x == 0.0 ? 1.0 : 0.0);
            }
            
            var comp = DiscreteChar.FromVector(res);
            return comp;
        }

        /// <summary>
        /// Gets a string representing this distribution.
        /// </summary>
        /// <param name="format">A format for the underlying probability vector.</param>
        /// <returns>A string representing this distribution.</returns>
        public override string ToString(string format)
        {
            return base.ToString(format, ",");
        }

        #endregion

        #region GenericDiscreteBase implementation

        /// <summary>
        /// Converts an integer to the corresponding character.
        /// </summary>
        /// <param name="value">The integer.</param>
        /// <returns>The character.</returns>
        protected override char ConvertFromInt(int value)
        {
            return (char)value;
        }

        /// <summary>
        /// Converts a character to the corresponding integer.
        /// </summary>
        /// <param name="value">The character.</param>
        /// <returns>The integer.</returns>
        protected override int ConvertToInt(char value)
        {
            return value;
        }

        /// <summary>
        /// Converts a given character to its string representation. Control characters are represented by their codes.
        /// </summary>
        /// <param name="ch">The character.</param>
        /// <returns>The string representing <paramref name="ch"/>.</returns>
        protected override string ToString(char ch)
        {
            return char.IsControl(ch) ? "#" + (int)ch : ch.ToString(CultureInfo.InvariantCulture);
        }

        #endregion
    }
}
