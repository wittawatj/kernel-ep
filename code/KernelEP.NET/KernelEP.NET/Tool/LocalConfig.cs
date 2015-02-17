using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

namespace KernelEP{

	// A local configuration file specific to a machine. 
	public class LocalConfig{

		// folder for saving files produced from code executions
		public const string K_SAVED_FOLDER = "saved_folder";
		public const string K_FACTOR_OP_MAT_FOLDER = "factor_op_mat_folder";
		// folder containing FactorOperator's in .mat files to be loaded into 
		// Infer.NET engine. A factor operator represents message operators for 
		// sending messages to all directions of a factor.
		public LocalConfig(){
		}

		public static Dictionary<string, string> GetLocalConfigs(){
			Dictionary<string, string> config = new Dictionary<string, string>();
			config.Add(K_SAVED_FOLDER, 
				"/nfs/nhome/live/wittawat/Dropbox/gatsby/research/code/KernelEP.NET/KernelEP.NET/Saved/");
			config.Add(K_FACTOR_OP_MAT_FOLDER, 
				"/nfs/nhome/live/wittawat/Dropbox/gatsby/research/code/saved/factor_op/");
			return config;
		}
	}
}

