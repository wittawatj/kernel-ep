/********************************************************
*                                                       *
*   Copyright (C) Microsoft. All rights reserved.       *
*                                                       *
********************************************************/

namespace MicrosoftResearch.Infer.Learners
{
    using System;

    /// <summary>
    /// Guards settings from being changed.
    /// </summary>
    [Serializable]
    public class SettingsGuard
    {
        /// <summary>
        /// A backward reference to the property indicating whether the setting is changeable or not.
        /// </summary>
        private readonly Func<bool> isImmutable;

        /// <summary>
        /// The message shown when trying to change an immutable setting.
        /// </summary>
        private readonly string message;

        /// <summary>
        /// Initializes a new instance of the <see cref="SettingsGuard"/> class.
        /// </summary>
        /// <param name="isImmutable">If true, the setting cannot be changed.</param>
        /// <param name="message">The message shown when trying to change an immutable setting.</param>
        public SettingsGuard(Func<bool> isImmutable, string message = "This setting cannot be changed.")
        {
            this.isImmutable = isImmutable;
            this.message = message;
        }

        /// <summary>
        /// Performs actions required before the value of a training setting is about to be changed.
        /// </summary>
        public void OnSettingChanging()
        {
            if (this.isImmutable())
            {
                throw new InvalidOperationException(this.message);
            }
        }
    }
}
