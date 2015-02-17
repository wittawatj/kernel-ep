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
    using System.Linq;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Distributions.Automata;
    using MicrosoftResearch.Infer.Utils;

    /// <summary>Provides outgoing messages for <see cref="Factor.StringFormat(String, String[])" />, given random arguments to the function.</summary>
    [FactorMethod(typeof(Factor), "StringFormat")]
    [Quality(QualityBand.Experimental)]
    public static class StringFormatOp
    {
        #region Fields and constants

        /// <summary>
        /// Used as a temporary replacement for '{' when computing messages. It is a unicode non-character.
        /// </summary>
        private const char LeftBraceReplacer = (char)0xFDD0;

        /// <summary>
        /// Used as a temporary replacement for '}' when computing messages. It is a unicode non-character.
        /// </summary>
        private const char RightBraceReplacer = (char)0xFDD1;

        /// <summary>
        /// For every possible number of arguments, stores transducers for escaping argument placeholders
        /// during message computation.
        /// </summary>
        private static readonly Dictionary<int, List<Pair<StringTransducer, StringTransducer>>> ArgumentCountToEscapingTransducers =
            new Dictionary<int, List<Pair<StringTransducer, StringTransducer>>>();

        /// <summary>
        /// A transducer used to assert that string cannot have
        /// <see cref="LeftBraceReplacer"/> or <see cref="RightBraceReplacer"/> inside it.
        /// </summary>
        private static readonly StringTransducer DisallowBraceReplacersTransducer;

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes static members of the <see cref="StringFormatOp"/> class.
        /// </summary>
        static StringFormatOp()
        {
            DiscreteChar noBraceReplacers = DiscreteChar.UniformOver(LeftBraceReplacer, RightBraceReplacer).Complement();
            DisallowBraceReplacersTransducer = StringTransducer.Copy(noBraceReplacers);
        }

        #endregion

        #region EP messages

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="format">Incoming message from <c>format</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(format,args) p(format,args) factor(str,format,args)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(StringDistribution format, IList<string> args)
        {
            return StrAverageConditional(format, ToAutomatonArray(args));
        }

        /// <summary>EP message to <c>str</c>.</summary>
        /// <param name="format">Incoming message from <c>format</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <returns>The outgoing EP message to the <c>str</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>str</c> as the random arguments are varied. The formula is <c>proj[p(str) sum_(format,args) p(format,args) factor(str,format,args)]/p(str)</c>.</para>
        /// </remarks>
        public static StringDistribution StrAverageConditional(StringDistribution format, IList<StringDistribution> args)
        {
            Argument.CheckIfNotNull(format, "format");
            Argument.CheckIfNotNull(args, "args");
            Argument.CheckIfValid(args.Count > 0, "args", "There must be at least one argument provided."); // TODO: relax?

            if (args.Count >= 10)
            {
                throw new NotImplementedException("Up to 10 arguments currently supported.");
            }
            
            // Disallow special characters in args or format
            StringAutomaton result = DisallowBraceReplacersTransducer.ProjectSource(format);
            List<StringAutomaton> escapedArgs = args.Select(DisallowBraceReplacersTransducer.ProjectSource).ToList();

            // Check braces for correctness and replace them with special characters.
            // Also, make sure that each argument placeholder is present exactly once.
            // Superposition of transducers is used instead of a single transducer to allow for any order of arguments.
            // TODO: in case of a single argument, argument escaping stage can be skipped
            for (int i = 0; i < args.Count; ++i)
            {
                result = GetArgumentEscapingTransducer(i, args.Count, false).ProjectSource(result);
            }

            // Now replace placeholders with arguments
            result = GetPlaceholderReplacingTransducer(escapedArgs, false).ProjectSource(result);
            return StringDistribution.FromWorkspace(result);
        }

        /// <summary>EP message to <c>format</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <returns>The outgoing EP message to the <c>format</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>format</c> as the random arguments are varied. The formula is <c>proj[p(format) sum_(str,args) p(str,args) factor(str,format,args)]/p(format)</c>.</para>
        /// </remarks>
        public static StringDistribution FormatAverageConditional(StringDistribution str, IList<string> args)
        {
            return FormatAverageConditional(str, ToAutomatonArray(args));
        }

        /// <summary>EP message to <c>format</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <returns>The outgoing EP message to the <c>format</c> argument.</returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>format</c> as the random arguments are varied. The formula is <c>proj[p(format) sum_(str,args) p(str,args) factor(str,format,args)]/p(format)</c>.</para>
        /// </remarks>
        public static StringDistribution FormatAverageConditional(StringDistribution str, IList<StringDistribution> args)
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfNotNull(args, "args");
            Argument.CheckIfValid(args.Count > 0, "args", "There must be at least one argument provided."); // TODO: relax?

            if (args.Count >= 10)
            {
                throw new NotImplementedException("Up to 10 arguments currently supported.");
            }

            // Disallow special characters in args
            List<StringAutomaton> escapedArgs = args.Select(DisallowBraceReplacersTransducer.ProjectSource).ToList();

            // Reverse the process defined by StrAverageConditional
            StringAutomaton result = GetPlaceholderReplacingTransducer(escapedArgs, true).ProjectSource(str);
            for (int i = 0; i < args.Count; ++i)
            {
                result = GetArgumentEscapingTransducer(i, args.Count, true).ProjectSource(result);
            }

            result = DisallowBraceReplacersTransducer.ProjectSource(result);
            return StringDistribution.FromWorkspace(result);
        }

        /// <summary>EP message to <c>args</c>.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <param name="format">Incoming message from <c>format</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <param name="result">Modified to contain the outgoing message.</param>
        /// <returns>
        ///   <paramref name="result" />
        /// </returns>
        /// <remarks>
        ///   <para>The outgoing message is a distribution matching the moments of <c>args</c> as the random arguments are varied. The formula is <c>proj[p(args) sum_(str,format) p(str,format) factor(str,format,args)]/p(args)</c>.</para>
        /// </remarks>
        /// <typeparam name="TStringDistributionList">The type of an outgoing message to <c>args</c>.</typeparam>
        public static TStringDistributionList ArgsAverageConditional<TStringDistributionList>(
            StringDistribution str, StringDistribution format, IList<StringDistribution> args, TStringDistributionList result)
            where TStringDistributionList : class, IList<StringDistribution>
        {
            Argument.CheckIfNotNull(str, "str");
            Argument.CheckIfNotNull(format, "format");
            Argument.CheckIfNotNull(args, "args");
            Argument.CheckIfValid(args.Count > 0, "args", "There must be at least one argument provided."); // TODO: relax?

            if (args.Count >= 10)
            {
                throw new NotImplementedException("Up to 10 arguments currently supported.");
            }
            
            var argsCopy = new List<StringDistribution>(args);
            for (int i = 0; i < args.Count; ++i)
            {
                argsCopy[i] = StringDistribution.Any();
                StringDistribution toStr = StrAverageConditional(format, argsCopy);
                StringDistribution toStrTimesStr = toStr.Product(str);
                toStrTimesStr.PruneToGroup((byte)(i + 1));
                result[i] = toStrTimesStr;
                argsCopy[i] = args[i];
            }

            return result;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="str">Incoming message from <c>str</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(str) p(str) factor(str,format,args) / sum_str p(str) messageTo(str))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        [Skip]
        public static double LogEvidenceRatio(StringDistribution str)
        {
            return 0;
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="str">Constant value for <c>str</c>.</param>
        /// <param name="format">Incoming message from <c>format</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(format,args) p(format,args) factor(str,format,args))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(string str, StringDistribution format, IList<StringDistribution> args)
        {
            StringDistribution toStr = StrAverageConditional(format, args);
            return toStr.GetLogProb(str);
        }

        /// <summary>Evidence message for EP.</summary>
        /// <param name="str">Constant value for <c>str</c>.</param>
        /// <param name="format">Incoming message from <c>format</c>.</param>
        /// <param name="args">Incoming message from <c>args</c>.</param>
        /// <returns>Logarithm of the factor's contribution the EP model evidence.</returns>
        /// <remarks>
        ///   <para>The formula for the result is <c>log(sum_(format,args) p(format,args) factor(str,format,args))</c>. Adding up these values across all factors and variables gives the log-evidence estimate for EP.</para>
        /// </remarks>
        public static double LogEvidenceRatio(string str, StringDistribution format, IList<string> args)
        {
            return LogEvidenceRatio(str, format, ToAutomatonArray(args));
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Creates a uniform distribution over all digits from 0 to <paramref name="maxDigit"/> (inclusive),
        /// excluding <paramref name="digit"/>.
        /// </summary>
        /// <param name="digit">The digit to exclude.</param>
        /// <param name="maxDigit">The maximum digit.</param>
        /// <returns>The created distribution.</returns>
        private static DiscreteChar AllDigitsExcept(int digit, int maxDigit)
        {
            Debug.Assert(maxDigit >= 0 && maxDigit <= 9 && digit >= 0 && digit <= 9, "The parameters must represent digits.");
            Debug.Assert(digit <= maxDigit, "digit must be between 0 and maxDigit.");
            Debug.Assert(maxDigit > 0, "The distribution must cover at least one digit.");
            
            if (digit == 0)
            {
                return DiscreteChar.InRange('1', (char)('0' + maxDigit));
            }

            if (digit == maxDigit)
            {
                return DiscreteChar.InRange('0', (char)('0' + maxDigit - 1));
            }

            return DiscreteChar.InRanges('0', (char)('0' + digit - 1), (char)('0' + digit + 1), (char)('0' + maxDigit));
        }

        /// <summary>
        /// Creates a transducer that replaces escaped argument placeholders with the corresponding arguments.
        /// </summary>
        /// <param name="escapedArgs">The list of arguments.</param>
        /// <param name="transpose">Specifies whether the created transducer should be transposed (useful for backward message computation).</param>
        /// <returns>The created transducer.</returns>
        private static StringTransducer GetPlaceholderReplacingTransducer(IList<StringAutomaton> escapedArgs, bool transpose)
        {
            var alternatives = new List<StringTransducer>();
            for (int i = 0; i < escapedArgs.Count; ++i)
            {
                StringTransducer alternative = StringTransducer.ConsumeElement((char)('0' + i));
                alternative.AppendInPlace(StringTransducer.Produce(escapedArgs[i]), (byte)(i + 1));
                alternatives.Add(alternative);
            }

            StringTransducer result = DisallowBraceReplacersTransducer.Clone();
            result.AppendInPlace(StringTransducer.ConsumeElement(LeftBraceReplacer));
            result.AppendInPlace(StringTransducer.Sum(alternatives));
            result.AppendInPlace(StringTransducer.ConsumeElement(RightBraceReplacer));
            result = StringTransducer.Repeat(result, minTimes: 0);
            result.AppendInPlace(DisallowBraceReplacersTransducer);

            if (transpose)
            {
                result.TransposeInPlace();
            }

            return result;
        }

        /// <summary>
        /// Creates a transducer that replaces braces surrounding the given argument placeholder
        /// with <see cref="LeftBraceReplacer"/> and <see cref="RightBraceReplacer"/>.
        /// </summary>
        /// <param name="argument">The index of the argument.</param>
        /// <param name="argumentCount">The total number of the arguments.</param>
        /// <param name="transpose">Specifies whether the created transducer should be transposed (useful for backward message computation).</param>
        /// <returns>The created transducer.</returns>
        private static StringTransducer GetArgumentEscapingTransducer(int argument, int argumentCount, bool transpose)
        {
            Debug.Assert(argumentCount >= 1 && argumentCount <= 10, "Up to 10 arguments currently supported.");
            Debug.Assert(argument >= 0 && argument < argumentCount, "Argument index must be less than the number of arguments.");

            List<Pair<StringTransducer, StringTransducer>> argumentToEscapingTransducers;
            if (!ArgumentCountToEscapingTransducers.TryGetValue(argumentCount, out argumentToEscapingTransducers))
            {
                argumentToEscapingTransducers = new List<Pair<StringTransducer, StringTransducer>>(argumentCount);
                ArgumentCountToEscapingTransducers[argumentCount] = argumentToEscapingTransducers;

                for (int i = 0; i < argumentCount; ++i)
                {
                    // Escapes braces in {i}
                    StringTransducer replaceBracesForDigit = StringTransducer.ConsumeElement('{');
                    replaceBracesForDigit.AppendInPlace(StringTransducer.ProduceElement(LeftBraceReplacer));
                    replaceBracesForDigit.AppendInPlace(StringTransducer.CopyElement((char)('0' + i)));
                    replaceBracesForDigit.AppendInPlace(StringTransducer.ConsumeElement('}'));
                    replaceBracesForDigit.AppendInPlace(StringTransducer.ProduceElement(RightBraceReplacer));

                    // Skips any number of placeholders which differ from {i}, with arbitrary intermediate text
                    DiscreteChar noBraces = DiscreteChar.UniformOver('{', '}').Complement();
                    StringTransducer braceReplacer = StringTransducer.Copy(noBraces);
                    if (argumentCount > 1)
                    {
                        // Skips every placeholder except {i}
                        StringTransducer skipOtherDigits = StringTransducer.CopyElement('{');
                        skipOtherDigits.AppendInPlace(StringTransducer.CopyElement(AllDigitsExcept(i, argumentCount - 1)));
                        skipOtherDigits.AppendInPlace(StringTransducer.CopyElement('}'));
                        
                        braceReplacer.AppendInPlace(skipOtherDigits);
                        braceReplacer = StringTransducer.Repeat(braceReplacer, minTimes: 0);
                        braceReplacer.AppendInPlace(StringTransducer.Copy(noBraces));    
                    }
                    
                    // Skips placeholders, then escapes {i} and skips placeholders
                    StringTransducer escapeAndSkip = replaceBracesForDigit.Clone();
                    escapeAndSkip.AppendInPlace(braceReplacer);
                    StringTransducer transducer = braceReplacer.Clone();
                    transducer.AppendInPlace(escapeAndSkip);  // TODO: use Optional() here if {i} can be omitted

                    StringTransducer transducerTranspose = StringTransducer.Transpose(transducer);
                    argumentToEscapingTransducers.Add(Pair.Create(transducer, transducerTranspose));
                }
            }

            return transpose ? argumentToEscapingTransducers[argument].Second : argumentToEscapingTransducers[argument].First;
        }

        /// <summary>
        /// Converts a list of string to an array of point mass automata.
        /// </summary>
        /// <param name="strings">The list of strings.</param>
        /// <returns>The created automata array.</returns>
        private static StringDistribution[] ToAutomatonArray(IList<string> strings)
        {
            return Util.ArrayInit(strings.Count, i => StringDistribution.PointMass(strings[i]));
        }

        #endregion
    }
}
