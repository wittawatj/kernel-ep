using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Windows.Forms;
using MicrosoftResearch.Infer.Tutorials.Views;
using MicrosoftResearch.Infer.Views;

namespace MicrosoftResearch.Infer.Tutorials
{
	public class TutorialViewer
	{
		[STAThread]
		public static void Main()
		{
			string path = Application.StartupPath;
			string[] tutorialFiles = Directory.GetFiles(path, "*.cs");
			TutorialsView tview = new TutorialsView();
			tview.TutorialFiles = tutorialFiles;
			Application.EnableVisualStyles();
			ModelView.RunInForm(tview, "Infer.NET tutorials viewer",false);
		}
	}
}
