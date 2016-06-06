using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class MainEnsemble
    {

        static void Main(string[] args)
        {
            string dirPath = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis";

            /*****************************************************
             * (0) Setup OpenCL
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    OpenCL setup");
            Console.WriteLine("=========================================\n");

            OpenCLSpace.SetupSpace(4);
            OpenCLSpace.KernelsPath = dirPath + "/ConvDotNet/Kernels";
            OpenCLSpace.LoadKernels();


            /*****************************************************
            * (1) Load data
            ******************************************************/

            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");

            // GTSRB test set (grayscale)
            DataSet testSetGS = new DataSet(43);
            string GTSRBtestDataGS = dirPath + "/GTSRB/Preprocessed/14_test_images.dat";
            string GTSRBtestLabelsGS = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";
            Console.WriteLine("Importing test set (grayscale)...");
            testSetGS.ReadData(GTSRBtestDataGS);
            testSetGS.ReadLabels(GTSRBtestLabelsGS);

            // GTSRB test set (RGB)
            DataSet testSetRGB = new DataSet(43);
            string GTSRBtestDataRGB = dirPath + "/GTSRB/Preprocessed/16_test_images.dat";
            string GTSRBtestLabelsRGB = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";
            Console.WriteLine("Importing test set (RGB)...");
            testSetRGB.ReadData(GTSRBtestDataRGB);
            testSetRGB.ReadLabels(GTSRBtestLabelsRGB);


            /*****************************************************
             * (2) Evaluate ensemble of networks
             *****************************************************/

            List<NeuralNetwork> networkEnsemble = new List<NeuralNetwork>();

            networkEnsemble.Add( Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_LeNet_GS_DropoutFC") );
            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_LeNet_RGB_DropoutFC"));
            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_VGGv2_GS_DropoutFC"));
            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_VGGv2_RGB_DropoutFC_reRun"));

            double error = 0.0;

            Console.WriteLine("\nEvaluating an ensemble of {0} networks...", networkEnsemble.Count);
            NetworkEvaluator.EvaluateEnsemble(networkEnsemble, testSetGS, testSetRGB, 64, out error);
            Console.WriteLine("\n\tTest set error = {0}\n\tAccuracy = {1}", error, 100*(1-error));


        }
    }
}
