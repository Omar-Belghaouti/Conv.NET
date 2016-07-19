using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace Conv.NET
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
            
            
            // GTSRB greyscale test set 1
            DataSet testSetGS1 = new DataSet(43);
            string GTSRBtestDataGS1 = dirPath + "/GTSRB/Preprocessed/14_test_images.dat";
            string GTSRBtestLabelsGS1 = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";
            Console.WriteLine("Importing test set (grayscale 1)...");
            testSetGS1.ReadData(GTSRBtestDataGS1, GTSRBtestLabelsGS1);
            
            /*
            // GTSRB greyscale test set 2
            DataSet testSetGS2 = new DataSet(43);
            string GTSRBtestDataGS2 = dirPath + "/GTSRB/Preprocessed/18_test_images.dat";
            string GTSRBtestLabelsGS2 = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";
            Console.WriteLine("Importing test set (grayscale 2)...");
            testSetGS2.ReadData(GTSRBtestDataGS2);
            testSetGS2.ReadLabels(GTSRBtestLabelsGS2);
            */
            
            // GTSRB RGB test set 1
            DataSet testSetRGB1 = new DataSet(43);
            string GTSRBtestDataRGB1 = dirPath + "/GTSRB/Preprocessed/16_test_images.dat";
            string GTSRBtestLabelsRGB1 = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";
            Console.WriteLine("Importing test set (RGB 1)...");
            testSetRGB1.ReadData(GTSRBtestDataRGB1, GTSRBtestLabelsRGB1);
            
            /*
            // GTSRB RGB test set 2
            DataSet testSetRGB2 = new DataSet(43);
            string GTSRBtestDataRGB2 = dirPath + "/GTSRB/Preprocessed/20_test_images.dat";
            string GTSRBtestLabelsRGB2 = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";
            Console.WriteLine("Importing test set (RGB 2)...");
            testSetRGB2.ReadData(GTSRBtestDataRGB2);
            testSetRGB2.ReadLabels(GTSRBtestLabelsRGB2);
            */

            /*****************************************************
             * (2) Evaluate ensemble of networks
             *****************************************************/

            List<NeuralNetwork> networkEnsemble = new List<NeuralNetwork>();

            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_LeNet_GS_DropoutFC"));
            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_LeNet_RGB_DropoutFC"));
            //networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "LeNet_GSb_DropoutFC"));
            //networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "LeNet_RGBb_Dropout"));
            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_VGGv2_GS_DropoutFC") );
            networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "FIXED_VGGv2_RGB_DropoutFC"));
            //networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "VGG_GSb_DropoutFC"));
            //networkEnsemble.Add(Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "VGG_RGBb_Dropout"));

            double error = 0.0;

            Console.WriteLine("\nEvaluating an ensemble of {0} networks...", networkEnsemble.Count);
            NetworkEvaluator.EvaluateEnsemble(networkEnsemble, testSetGS1, testSetRGB1, 64, out error);
            Console.WriteLine("\n\tTest set error = {0}\n\tAccuracy = {1}", error, 100*(1-error));


        }
    }
}
