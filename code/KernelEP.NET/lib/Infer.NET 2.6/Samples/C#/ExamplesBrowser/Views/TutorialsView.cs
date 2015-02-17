using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Reflection;
using System.Drawing;
using System.Text.RegularExpressions;
using System.ComponentModel;

namespace MicrosoftResearch.Infer.Tutorials.Views
{
	/// <summary>
	/// A view of a set of tutorials or examples which shows both source code and tutorial output.
	/// </summary>
	class TutorialsView : UserControl
	{
		private SplitContainer splitContainer1;
		private TreeView tutorialsTree;
		private SplitContainer splitContainer2;
		private RichTextBox sourceTextBox;
		private RichTextBox outputTextBox;
		private Panel panel2;
		private Button button1;
		private BackgroundWorker worker;
		private RichTextBox exampleSummaryBox;
		private Panel panel1;

		/// <summary>
		/// Creates a tutorial view.
		/// </summary>
		public TutorialsView()
		{
			InitializeComponent();
			sourceTextBox.SelectionTabs = new int[] { 20,40,60,80,100,120};
			tutorialsTree.AfterSelect += new TreeViewEventHandler(tutorialsTree_AfterSelect);
		}

		/// <summary>
		/// Creates a view of the tutorials in the specified namespace.
		/// </summary>
		/// <param name="namespce">The namespace to search to find tutorials</param>
		public TutorialsView(string namespce) : this()
		{
			Namespace = namespce;
		}

		void tutorialsTree_AfterSelect(object sender, TreeViewEventArgs e)
		{
			OnSelectionChanged();
		}

		/// <summary>
		/// The currently selected tutorial.
		/// </summary>
		public Type SelectedTutorial
		{
			get
			{
				TreeNode nd = tutorialsTree.SelectedNode;
				if ((nd == null) || (!(nd.Tag is Type))) return null;
				return (Type)nd.Tag;
			}
		}

		protected void OnSelectionChanged()
		{
			Type tutorialClass = SelectedTutorial;
			if (tutorialClass == null) return;
			exampleSummaryBox.Clear();
			exampleSummaryBox.SelectionColor = Color.DarkBlue;
			exampleSummaryBox.SelectionFont = new Font(FontFamily.GenericSansSerif, 11f,FontStyle.Bold);
			exampleSummaryBox.AppendText(tutorialClass.Name+Environment.NewLine);
			//label1.Text = "'" + tutorialClass.Name + "' source code";
			ExampleAttribute exa = GetExampleAttribute(tutorialClass);
			exampleSummaryBox.SelectionFont = new Font(FontFamily.GenericSansSerif, 10f, FontStyle.Regular);
			exampleSummaryBox.SelectionColor = Color.FromArgb(0, 0, 100);

			string desc = "";
			if (exa != null) desc = exa.Description;
			exampleSummaryBox.AppendText(desc);
			exampleSummaryBox.Refresh();
			string filename = GetSourceCodeFilename(tutorialClass);
			sourceTextBox.Clear();
			if (filename == null)
			{
				sourceTextBox.AppendText("Tutorial source code was not found.  "+Environment.NewLine+
				"Go to the properties of the source file in Visual Studio and set 'Copy To Output Directory' to 'Copy if newer'.");
				return;
			}
			StreamReader sr = new StreamReader(filename);
			while (true)
			{
				string line = sr.ReadLine();
				if (line == null) break;
				PrettyPrint(line);
			}
			sr.Close();
			this.PerformLayout();
		}

		Regex reg = new Regex(@"[\w]+");
		/// <summary>
		/// Very simple syntax highlighting
		/// </summary>
		/// <param name="s"></param>
		protected void PrettyPrint(string s)
		{
			if (s.Trim().StartsWith("[Example(")) return;
			if (s.Trim().StartsWith("//"))
			{
				sourceTextBox.SelectionColor = Color.Green;
				sourceTextBox.AppendText(s + Environment.NewLine);
				sourceTextBox.SelectionColor = Color.Black;
				return;
			}
			MatchCollection mc = reg.Matches(s);
			int ind = 0;
			foreach (Match m in mc)
			{
				sourceTextBox.AppendText(s.Substring(ind,m.Index-ind));
				ind = m.Index + m.Length;
				string word = s.Substring(m.Index , m.Length);
				bool reserved = IsReservedWord(word);
				if (reserved) sourceTextBox.SelectionColor = Color.Blue;
				sourceTextBox.AppendText(word);
				sourceTextBox.SelectionColor = Color.Black;
			}
			if (ind<s.Length) sourceTextBox.AppendText(s.Substring(ind));
			sourceTextBox.AppendText(Environment.NewLine);
			//sourceTextBox.AppendText(s + Environment.NewLine);
		}

		private Dictionary<string, bool> reservedSet = new Dictionary<string, bool>();
		private static string[] RESERVED_WORDS = { "abstract","as","base","bool","break","byte","case","catch","char","checked",
		"class","const","continue","decimal","default","delegate","do","double","else","enum","event","explicit","extern","false",
		"finally","fixed","float","for","foreach","goto","if","implicit","in","int","interface","internal","is","lock","long","namespace",
		"new","null","object","operator","out","override","params","private","protected","public","readonly","ref","return","sbyte",
		"sealed","short","sizeof","stackalloc","static","string","struct","switch","this","throw","true","try","typeof","uint","ulong",
		"unchecked","unsafe","ushort","using","virtual","volatile","void","while"};
		private bool IsReservedWord(string word)
		{
			if (reservedSet.Count == 0)
			{
				foreach (string s in RESERVED_WORDS) reservedSet[s] = true;
			}
			return reservedSet.ContainsKey(word);
		}

		private string GetSourceCodeFilename(Type tutorialClass)
		{
			string filename = tutorialClass.Name+".cs";
			if (File.Exists(filename)) return filename;
			 filename = "../../"+tutorialClass.Name + ".cs";
			if (File.Exists(filename)) return filename;
			return null;
		}

		private void RunSelectedTutorial()
		{
			Type tp = SelectedTutorial;
			if (tp == null) return;
			button1.Enabled = false;
			button1.Text = tp.Name+" running...";
			button1.Refresh();
			outputTextBox.Clear();
			outputTextBox.Refresh();
			worker.RunWorkerAsync(tp);
		}

		private void worker_DoWork(object sender, DoWorkEventArgs e)
		{
			Type tp = (Type)e.Argument;
			TextWriter tw = Console.Out;
			Console.SetOut(new GUITextWriter(this));
			RunMe.RunTutorial(tp);
			Console.SetOut(tw);
		}

		private void worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			button1.Enabled = true;
			button1.Text = "Run this tutorial";
		}

		protected string namespce = "MicrosoftResearch.Infer.Tutorials";
		/// <summary>
		/// The namespace to search to find tutorials.
		/// </summary>
		public string Namespace
		{
			get { return namespce; }
			set { namespce = value; OnTutorialNamespaceChanged(); }
		}


		/// <summary>
		/// Called when the tutorials namespace changes
		/// </summary>
		protected void OnTutorialNamespaceChanged()
		{
			Type[] types = typeof(TutorialsView).Assembly.GetTypes();
			tutorialsTree.Nodes.Clear();
			tutorialsTree.ShowNodeToolTips = true;
			foreach (Type t in types)
			{
				if (t.Namespace != namespce) continue;
				if (t.GetMethod("Run") == null) continue;
				string category = "Examples";
				ExampleAttribute exa = GetExampleAttribute(t);
				if (exa != null) category = exa.Category;
				TreeNode par = null;
				foreach (TreeNode nd in tutorialsTree.Nodes)
				{
					if (category.Equals(nd.Tag)) par = nd;
				}
				if (par == null)
				{
					par = tutorialsTree.Nodes.Add(category);
					par.Tag = category;
					par.NodeFont = new Font(tutorialsTree.Font, FontStyle.Bold);
					par.Text = category;
				}
				string name = t.Name;
				TreeNode nd2 = par.Nodes.Add(name);
				if (exa != null) nd2.ToolTipText = exa.Description;
				nd2.Tag = t;
			}
			tutorialsTree.ExpandAll();
			if ((tutorialsTree.Nodes.Count > 0) && (tutorialsTree.Nodes[0].Nodes.Count>0))
			{
				tutorialsTree.SelectedNode = tutorialsTree.Nodes[0].Nodes[0];
			}
		}

		private ExampleAttribute GetExampleAttribute(Type exampleClass)
		{
			object[] arr = exampleClass.GetCustomAttributes(typeof(ExampleAttribute), true);
			if ((arr != null) && (arr.Length > 0)) return ((ExampleAttribute)arr[0]);
			return null;
		}

		private void InitializeComponent()
		{
			this.splitContainer1 = new System.Windows.Forms.SplitContainer();
			this.tutorialsTree = new System.Windows.Forms.TreeView();
			this.splitContainer2 = new System.Windows.Forms.SplitContainer();
			this.sourceTextBox = new System.Windows.Forms.RichTextBox();
			this.panel1 = new System.Windows.Forms.Panel();
			this.exampleSummaryBox = new System.Windows.Forms.RichTextBox();
			this.outputTextBox = new System.Windows.Forms.RichTextBox();
			this.panel2 = new System.Windows.Forms.Panel();
			this.button1 = new System.Windows.Forms.Button();
			this.worker = new System.ComponentModel.BackgroundWorker();
			this.splitContainer1.Panel1.SuspendLayout();
			this.splitContainer1.Panel2.SuspendLayout();
			this.splitContainer1.SuspendLayout();
			this.splitContainer2.Panel1.SuspendLayout();
			this.splitContainer2.Panel2.SuspendLayout();
			this.splitContainer2.SuspendLayout();
			this.panel1.SuspendLayout();
			this.panel2.SuspendLayout();
			this.SuspendLayout();
			// 
			// splitContainer1
			// 
			this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
			this.splitContainer1.Location = new System.Drawing.Point(0, 0);
			this.splitContainer1.Name = "splitContainer1";
			// 
			// splitContainer1.Panel1
			// 
			this.splitContainer1.Panel1.Controls.Add(this.tutorialsTree);
			// 
			// splitContainer1.Panel2
			// 
			this.splitContainer1.Panel2.Controls.Add(this.splitContainer2);
			this.splitContainer1.Size = new System.Drawing.Size(853, 493);
			this.splitContainer1.SplitterDistance = 170;
			this.splitContainer1.TabIndex = 0;
			// 
			// tutorialsTree
			// 
			this.tutorialsTree.Dock = System.Windows.Forms.DockStyle.Fill;
			this.tutorialsTree.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
			this.tutorialsTree.HideSelection = false;
			this.tutorialsTree.Location = new System.Drawing.Point(0, 0);
			this.tutorialsTree.Name = "tutorialsTree";
			this.tutorialsTree.Size = new System.Drawing.Size(170, 493);
			this.tutorialsTree.TabIndex = 0;
			// 
			// splitContainer2
			// 
			this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
			this.splitContainer2.Location = new System.Drawing.Point(0, 0);
			this.splitContainer2.Name = "splitContainer2";
			this.splitContainer2.Orientation = System.Windows.Forms.Orientation.Horizontal;
			// 
			// splitContainer2.Panel1
			// 
			this.splitContainer2.Panel1.Controls.Add(this.sourceTextBox);
			this.splitContainer2.Panel1.Controls.Add(this.panel1);
			// 
			// splitContainer2.Panel2
			// 
			this.splitContainer2.Panel2.Controls.Add(this.outputTextBox);
			this.splitContainer2.Panel2.Controls.Add(this.panel2);
			this.splitContainer2.Size = new System.Drawing.Size(679, 493);
			this.splitContainer2.SplitterDistance = 335;
			this.splitContainer2.TabIndex = 0;
			// 
			// sourceTextBox
			// 
			this.sourceTextBox.BackColor = System.Drawing.SystemColors.ControlLightLight;
			this.sourceTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
			this.sourceTextBox.Font = new System.Drawing.Font("Courier New", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
			this.sourceTextBox.Location = new System.Drawing.Point(0, 40);
			this.sourceTextBox.Name = "sourceTextBox";
			this.sourceTextBox.ReadOnly = true;
			this.sourceTextBox.ShowSelectionMargin = true;
			this.sourceTextBox.Size = new System.Drawing.Size(679, 295);
			this.sourceTextBox.TabIndex = 3;
			this.sourceTextBox.Text = "";
			this.sourceTextBox.WordWrap = false;
			// 
			// panel1
			// 
			this.panel1.Controls.Add(this.exampleSummaryBox);
			this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
			this.panel1.Location = new System.Drawing.Point(0, 0);
			this.panel1.Name = "panel1";
			this.panel1.Size = new System.Drawing.Size(679, 40);
			this.panel1.TabIndex = 2;
			// 
			// exampleSummaryBox
			// 
			this.exampleSummaryBox.BackColor = System.Drawing.SystemColors.Control;
			this.exampleSummaryBox.Dock = System.Windows.Forms.DockStyle.Fill;
			this.exampleSummaryBox.Location = new System.Drawing.Point(0, 0);
			this.exampleSummaryBox.Margin = new System.Windows.Forms.Padding(8, 3, 8, 3);
			this.exampleSummaryBox.Name = "exampleSummaryBox";
			this.exampleSummaryBox.ReadOnly = true;
			this.exampleSummaryBox.ShowSelectionMargin = true;
			this.exampleSummaryBox.Size = new System.Drawing.Size(679, 40);
			this.exampleSummaryBox.TabIndex = 2;
			this.exampleSummaryBox.Text = "";
			// 
			// outputTextBox
			// 
			this.outputTextBox.Dock = System.Windows.Forms.DockStyle.Fill;
			this.outputTextBox.Font = new System.Drawing.Font("Courier New", 9F);
			this.outputTextBox.Location = new System.Drawing.Point(0, 32);
			this.outputTextBox.Name = "outputTextBox";
			this.outputTextBox.ReadOnly = true;
			this.outputTextBox.Size = new System.Drawing.Size(679, 122);
			this.outputTextBox.TabIndex = 4;
			this.outputTextBox.Text = "";
			// 
			// panel2
			// 
			this.panel2.Controls.Add(this.button1);
			this.panel2.Dock = System.Windows.Forms.DockStyle.Top;
			this.panel2.Location = new System.Drawing.Point(0, 0);
			this.panel2.Name = "panel2";
			this.panel2.Size = new System.Drawing.Size(679, 32);
			this.panel2.TabIndex = 3;
			// 
			// button1
			// 
			this.button1.Dock = System.Windows.Forms.DockStyle.Fill;
			this.button1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
			this.button1.Location = new System.Drawing.Point(0, 0);
			this.button1.Name = "button1";
			this.button1.Size = new System.Drawing.Size(679, 32);
			this.button1.TabIndex = 2;
			this.button1.Text = "Run this tutorial";
			this.button1.UseVisualStyleBackColor = true;
			this.button1.Click += new System.EventHandler(this.button1_Click);
			// 
			// worker
			// 
			this.worker.DoWork += new System.ComponentModel.DoWorkEventHandler(this.worker_DoWork);
			this.worker.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.worker_RunWorkerCompleted);
			// 
			// TutorialsView
			// 
			this.Controls.Add(this.splitContainer1);
			this.Name = "TutorialsView";
			this.Size = new System.Drawing.Size(853, 493);
			this.splitContainer1.Panel1.ResumeLayout(false);
			this.splitContainer1.Panel2.ResumeLayout(false);
			this.splitContainer1.ResumeLayout(false);
			this.splitContainer2.Panel1.ResumeLayout(false);
			this.splitContainer2.Panel2.ResumeLayout(false);
			this.splitContainer2.ResumeLayout(false);
			this.panel1.ResumeLayout(false);
			this.panel2.ResumeLayout(false);
			this.ResumeLayout(false);

		}

		private void button1_Click(object sender, EventArgs e)
		{
			RunSelectedTutorial();
		}

		public void AppendOutputText(string s)
		{
			outputTextBox.AppendText(s);
			outputTextBox.ScrollToCaret();
		}

	}

	/// <summary>
	/// Used to redirect console output to a text box
	/// </summary>
	class GUITextWriter : StringWriter
	{
		TutorialsView view;

		public delegate void UpdateTextCallback(string text);

		internal GUITextWriter(TutorialsView view)
		{
			this.view=view;
		}

		public override void Write(string value)
		{
			base.Write(value);
			view.Invoke(new UpdateTextCallback(view.AppendOutputText), value);
		}

		public override void WriteLine(string value)
		{
			Write(value+Environment.NewLine);
		}
	}
}
