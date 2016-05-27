using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class JaNetProgram
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

            OpenCLSpace.SetupSpace();
            OpenCLSpace.KernelsPath = dirPath + "/ConvDotNet/Kernels";
            OpenCLSpace.LoadKernels();


            /*****************************************************
            * (1) Load data
            ******************************************************/

            string imageColor = "GS";

            #region DataImport

            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");

            DataSet trainingSet = new DataSet(43);
            DataSet validationSet = new DataSet(43);
            DataSet testSet = new DataSet(43);

            if (imageColor == "GS")
            {
                // GTSRB training set (grayscale)
                string GTSRBtrainingDataGS = dirPath + "/GTSRB/Preprocessed/14_training_images.dat";
                string GTSRBtrainingLabelsGS = dirPath + "/GTSRB/Preprocessed/14_training_classes.dat";


                // GTSRB validation set (grayscale)
                string GTSRBvalidationDataGS = dirPath + "/GTSRB/Preprocessed/14_validation_images.dat";
                string GTSRBvalidationLabelsGS = dirPath + "/GTSRB/Preprocessed/14_validation_classes.dat";


                // GTSRB test set (grayscale)
                string GTSRBtestDataGS = dirPath + "/GTSRB/Preprocessed/14_test_images.dat";
                string GTSRBtestLabelsGS = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";

                Console.WriteLine("Importing training set...");
                trainingSet.ReadData(GTSRBtrainingDataGS);
                trainingSet.ReadLabels(GTSRBtrainingLabelsGS);


                Console.WriteLine("Importing validation set...");
                validationSet.ReadData(GTSRBvalidationDataGS);
                validationSet.ReadLabels(GTSRBvalidationLabelsGS);


                Console.WriteLine("Importing test set...");
                testSet.ReadData(GTSRBtestDataGS);
                testSet.ReadLabels(GTSRBtestLabelsGS);
            }
            else if (imageColor == "RGB")
            {
                // GTSRB training set (RGB)
                string GTSRBtrainingDataRGB = dirPath + "/GTSRB/Preprocessed/16_training_images.dat";
                string GTSRBtrainingLabelsRGB = dirPath + "/GTSRB/Preprocessed/16_training_classes.dat";


                // GTSRB validation set (RGB)
                string GTSRBvalidationDataRGB = dirPath + "/GTSRB/Preprocessed/16_validation_images.dat";
                string GTSRBvalidationLabelsRGB = dirPath + "/GTSRB/Preprocessed/16_validation_classes.dat";


                // GTSRB test set (RGB)
                string GTSRBtestDataRGB = dirPath + "/GTSRB/Preprocessed/16_test_images.dat";
                string GTSRBtestLabelsRGB = dirPath + "/GTSRB/Preprocessed/test_labels_full.dat";

                Console.WriteLine("Importing training set...");
                trainingSet.ReadData(GTSRBtrainingDataRGB);
                trainingSet.ReadLabels(GTSRBtrainingLabelsRGB);


                Console.WriteLine("Importing validation set...");
                validationSet.ReadData(GTSRBvalidationDataRGB);
                validationSet.ReadLabels(GTSRBvalidationLabelsRGB);


                Console.WriteLine("Importing test set...");
                testSet.ReadData(GTSRBtestDataRGB);
                testSet.ReadLabels(GTSRBtestLabelsRGB);

            }
            #endregion

            /*****************************************************
             * (2) Instantiate a neural network and add layers
             * 
             * OPTIONS:
             * ConvolutionalLayer(filterSize, numberOfFilters, strideLength, zeroPadding)
             * ResidualModule(filterSize, numberOfFilters, strideLength, zeroPadding, nonlinearityType)
             * FullyConnectedLayer(numberOfUnits)
             * MaxPooling(2, 2)
             * AveragePooling()
             * ReLU()
             * ELU(alpha)
             * SoftMax()
             ****************************************************/

            #region NeuralNetworkCreation

            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Neural network creation");
            Console.WriteLine("=========================================\n");

            // OPTION 1: Create a new network
            
            NeuralNetwork network = new NeuralNetwork("test_simpLeNet_HighWeightDecay");

            network.AddLayer(new InputLayer(1, 32, 32));

            network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ResidualModule(3, 32, 1, 1, "ELU"));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ResidualModule(3, 32, 1, 1, "ELU"));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ConvolutionalLayer(3, 64, 2, 1));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ResidualModule(3, 64, 1, 1, "ELU"));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ResidualModule(3, 64, 1, 1, "ELU"));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ConvolutionalLayer(3, 128, 2, 1));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new ResidualModule(3, 128, 1, 1, "ELU"));
            network.AddLayer(new ELU(1.0f));

            network.AddLayer(new AveragePooling());

            network.AddLayer(new FullyConnectedLayer(43));
            network.AddLayer(new SoftMax());

            NetworkTrainer.TrainingMode = "new";
            

            // OPTION 2: Load a network from file
            /*
            NeuralNetwork network = Utils.LoadNetworkFromFile(dirPath + "/Results/Networks/", "ResNetV2_GS_ROIS_DropoutConv");
            //network.Name = "VGG_GS_ROIS_NoDropout";
            network.Set("MiniBatchSize", 64); // this SHOULDN'T matter!
            network.InitializeParameters("load");
            NetworkTrainer.TrainingMode = "resume";
            */

            #endregion


            /*****************************************************
            * (3) Gradient check
            ******************************************************/
            //GradientChecker.Check(network, validationSet);
            
            
            /*****************************************************
            * (4) Train network
            ******************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");

            #region Training

            // Set output files save paths
            string trainingSavePath = dirPath + "/Results/LossError/";
            NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
            NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
            NetworkTrainer.NetworkOutputFilePath = dirPath + "/Results/Networks/";

            NetworkTrainer.MomentumMultiplier = 0.9;
            NetworkTrainer.WeightDecayCoeff = 0.0;
            NetworkTrainer.MaxTrainingEpochs = 200;
            NetworkTrainer.EpochsBeforeRegularization = 0;
            NetworkTrainer.MiniBatchSize = 64;
            NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
            NetworkTrainer.EvaluateBeforeTraining = true;
            NetworkTrainer.DropoutFullyConnected = 1.0;
            NetworkTrainer.DropoutConvolutional = 1.0;
            NetworkTrainer.DropoutInput = 1.0;
            NetworkTrainer.Patience = 10;
            NetworkTrainer.LearningRateDecayFactor = Math.Sqrt(10.0);
            NetworkTrainer.MaxConsecutiveAnnealings = 3;

            NetworkTrainer.LearningRate = 0.01;
            NetworkTrainer.Train(network, validationSet, null);

            #endregion

            /*****************************************************
             * (5) Test network
             *****************************************************/

            # region Testing
            /*
            Console.WriteLine("\nFINAL EVALUATION:");

            // Load best network from file
            NeuralNetwork bestNetwork = Utils.LoadNetworkFromFile("../../../../Results/Networks/", network.Name);
            bestNetwork.Set("MiniBatchSize", 64); // this SHOULDN'T matter!
            bestNetwork.InitializeParameters("load");
            bestNetwork.Set("Inference", true);

            double loss;
            double error;

            //NetworkEvaluator.EvaluateNetwork(bestNetwork, trainingSet, out loss, out error);
            //Console.WriteLine("\nTraining set:\n\tLoss = {0}\n\tError = {1}", loss, error);

            NetworkEvaluator.EvaluateNetwork(bestNetwork, validationSet, out loss, out error);
            Console.WriteLine("\nValidation set:\n\tLoss = {0}\n\tError = {1}", loss, error);

            NetworkEvaluator.EvaluateNetwork(bestNetwork, testSet, out loss, out error);
            Console.WriteLine("\nTest set:\n\tLoss = {0}\n\tError = {1}", loss, error);

            // Save misclassified examples
            //NetworkEvaluator.SaveMisclassifiedExamples(bestNetwork, trainingSet, "../../../../Results/MisclassifiedExamples/" + network.Name + "_training.txt");
            //NetworkEvaluator.SaveMisclassifiedExamples(bestNetwork, validationSet, "../../../../Results/MisclassifiedExamples/" + network.Name + "_validation.txt");
            //NetworkEvaluator.SaveMisclassifiedExamples(bestNetwork, testSet, "../../../../Results/MisclassifiedExamples/" + network.Name + "_test.txt");

            // Save filters of first conv layer
            //if (bestNetwork.Layers[1].Type == "Convolutional")
            //    Utils.SaveFilters(bestNetwork, "../../../../Results/Filters/" + network.Name + "_filters.txt");
            */
            #endregion

            /*****************************************************/
        }
    }
}