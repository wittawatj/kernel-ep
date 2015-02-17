using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Windows.Forms;
using MicrosoftResearch.Infer.Views;
using System.Reflection;

namespace MicrosoftResearch.Infer.Tutorials
{
	/// <summary>
	/// Use this class to run tutorials or to show the tutorial viewer.
	/// </summary>
	public class RunMe
	{
		[STAThread]
		public static void Main()
		{
			Type tutorialClass = null;
			//********** UNCOMMENT AND EDIT THIS LINE TO RUN A PARTICULAR TUTORIAL DIRECTLY *************
			//tutorialClass = typeof(MicrosoftResearch.Infer.Tutorials.FirstExample);

			if (tutorialClass != null)
			{
				// Run the specified tutorial
				RunTutorial(tutorialClass);
			}
			else
			{
				// Show all tutorials, in a browser
				IAlgorithm[] algs = InferenceEngine.GetBuiltInAlgorithms();
				// Avoid max product in the examples browser, as none of the examples apply.
				List<IAlgorithm> algList = new List<IAlgorithm>(algs);
				algList.RemoveAll(alg => alg is MaxProductBeliefPropagation);
				ExamplesViewer tview = new ExamplesViewer(typeof(RunMe), algList.ToArray());
				tview.Run();
			}
		}

		/// <summary>
		/// Runs the tutorial contained in the supplied class.
		/// </summary>
		/// <param name="tutorialClass">The class containing the tutorial to be run</param>
		public static void RunTutorial(Type tutorialClass)
		{
			if (tutorialClass == null) return;
			object obj = Activator.CreateInstance(tutorialClass);
			MethodInfo mi = tutorialClass.GetMethod("Run");
			if (mi == null) return;
			mi.Invoke(obj, new object[0]);
		}
	}

}
