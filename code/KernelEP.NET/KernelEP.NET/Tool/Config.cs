using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;

using System.IO;

namespace KernelEP{
	// Global config of kernel EP
	public class Config{
		public Config(){
		}

		public static string PathTo(string key, string fileName){
			Dictionary<string, string> config = LocalConfig.GetLocalConfigs();
			string folder = config[key];
			string fullPath = Path.Combine(folder, fileName);
			return fullPath;
		}
		// construct a full path to the file name in the "saved" folder.
		public static string PathToSavedFile(string fileName){
			return PathTo(LocalConfig.K_SAVED_FOLDER, fileName);

		}

		// construct a full path to a file name in the factor_op folder.
		// See LocalConfig.
		public static string PathToFactorOperator(string fileName){
			return PathTo(LocalConfig.K_FACTOR_OP_MAT_FOLDER, fileName);
		}

		public static string PathToCompiledModelFolder(){
			Dictionary<string, string> config = LocalConfig.GetLocalConfigs();
			string folder = config[LocalConfig.K_COMPILED_MODEL_FOLDER];
			return folder;

		}

	}
}

