
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

            /*****************************************************
             * (0) Setup OpenCL
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    OpenCL setup");
            Console.WriteLine("=========================================\n");

            OpenCLSpace.SetupSpace();
            OpenCLSpace.KernelsPath = "../../../Kernels";
            OpenCLSpace.LoadKernels();


                // OPTION 2: Load a network from file
            /*
                NeuralNetwork network = Utils.LoadNetworkFromFile(@"C:\Users\jacopo\Dropbox\Chalmers\MSc thesis\Results\Networks\", "NetworkName");
                network.Set("MiniBatchSize", 64); // this SHOULDN'T matter!
                network.InitializeParameters("load");
                NetworkTrainer.TrainingMode = "resume";
                */

                /*****************************************************
                    * (2) Load data
                    ****************************************************/
                Console.WriteLine("\n=========================================");
                Console.WriteLine("    Importing data");
                Console.WriteLine("=========================================\n");

               
                // GTSRB training set
                string GTSRBtrainingDataGS = "../../../../GTSRB/Preprocessed/12_training_images.dat";
                string GTSRBtrainingLabelsGS = "../../../../GTSRB/Preprocessed/12_training_classes.dat";
            

                // GTSRB validation set (grayscale)
            
                string GTSRBvalidationDataGS = "../../../../GTSRB/Preprocessed/12_validation_images.dat";
                string GTSRBvalidationLabelsGS = "../../../../GTSRB/Preprocessed/12_validation_classes.dat";

            
                // GTSRB test set (grayscale)
                string GTSRBtestDataGS = "../../../../GTSRB/Preprocessed/12_test_images.dat";
                string GTSRBtestLabelsGS = "../../../../GTSRB/Preprocessed/test_labels_full.dat";
            
                
                Console.WriteLine("Importing training set...");
                DataSet trainingSet = new DataSet(43);
                trainingSet.ReadData(GTSRBtrainingDataGS);
                trainingSet.ReadLabels(GTSRBtrainingLabelsGS);
                
            
                Console.WriteLine("Importing validation set...");
                DataSet validationSet = new DataSet(43);
                validationSet.ReadData(GTSRBvalidationDataGS);
                validationSet.ReadLabels(GTSRBvalidationLabelsGS);
            
                
                Console.WriteLine("Importing test set...");
                DataSet testSet = new DataSet(43);
                testSet.ReadData(GTSRBtestDataGS);
                testSet.ReadLabels(GTSRBtestLabelsGS);

                /*****************************************************
                 * (1) Instantiate a neural network and add layers
                 * 
                 * OPTIONS:
                 * ConvolutionalLayer(filterSize, numberOfFilters, strideLength, zeroPadding)
                 * FullyConnectedLayer(numberOfUnits)
                 * MaxPooling(2, 2)
                 * ReLU()
                 * ELU(alpha)
                 * SoftMax()
                 ****************************************************/

                double[] eta = {0.00001};

                
                // LENET_RELU ____________________________________________________________________________________________________
            /*
                for (int iEta = 0; iEta < eta.Length; iEta++)
                {
                    Console.WriteLine("\n\n\n New learning rate = {0}", eta[iEta]);



                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Neural network creation");
                    Console.WriteLine("=========================================\n");

                    // OPTION 1: Create a new network

                    NeuralNetwork network = new NeuralNetwork("EtaTest_LeNet_ReLU");

                    network.AddLayer(new InputLayer(1, 32, 32));

                    network.AddLayer(new ConvolutionalLayer(5, 108, 1, 0));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new ConvolutionalLayer(5, 108, 1, 0));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new FullyConnectedLayer(100));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new FullyConnectedLayer(100));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new FullyConnectedLayer(43));
                    network.AddLayer(new SoftMax());

                    NetworkTrainer.TrainingMode = "new";


                /*****************************************************
                    * (3) Gradient check
                    ****************\*********************************
                GradientChecker.Check(network, validationSet);





                /*****************************************************
                    * (4) Train network
                    *****************************************************
                Console.WriteLine("\n=========================================");
                Console.WriteLine("    Network training");
                Console.WriteLine("=========================================\n");

                // Set output files save paths
                string trainingSavePath = "../../../../Results/LossError/";
                NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
                NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
                NetworkTrainer.NetworkOutputFilePath = "../../../../Results/Networks/";

                NetworkTrainer.MomentumMultiplier = 0.9;
                NetworkTrainer.WeightDecayCoeff = 0.0;
                NetworkTrainer.MaxTrainingEpochs = 1;
                NetworkTrainer.EpochsBeforeRegularization = 0;
                NetworkTrainer.MiniBatchSize = 64;
                NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
                NetworkTrainer.EvaluateBeforeTraining = true;
                NetworkTrainer.DropoutFullyConnected = 1.0;
                NetworkTrainer.DropoutConvolutional = 1.0;
                NetworkTrainer.Patience = 20;

                NetworkTrainer.LearningRate = eta[iEta];
                NetworkTrainer.Train(network, trainingSet, null);
            }



                // LENET_ELU ____________________________________________________________________________________________________

                for (int iEta = 0; iEta < eta.Length; iEta++)
                {
                    Console.WriteLine("\n\n\n New learning rate = {0}", eta[iEta]);



                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Neural network creation");
                    Console.WriteLine("=========================================\n");

                    // OPTION 1: Create a new network

                    NeuralNetwork network = new NeuralNetwork("EtaTest_LeNet_ELU");

                    network.AddLayer(new InputLayer(1, 32, 32));

                    network.AddLayer(new ConvolutionalLayer(5, 108, 1, 0));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new ConvolutionalLayer(5, 108, 1, 0));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new FullyConnectedLayer(100));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new FullyConnectedLayer(100));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new FullyConnectedLayer(43));
                    network.AddLayer(new SoftMax());

                    NetworkTrainer.TrainingMode = "new";


                    /*****************************************************
                        * (3) Gradient check
                        ****************\*********************************
                    GradientChecker.Check(network, validationSet);





                    /*****************************************************
                        * (4) Train network
                        *****************************************************
                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Network training");
                    Console.WriteLine("=========================================\n");

                    // Set output files save paths
                    string trainingSavePath = "../../../../Results/LossError/";
                    NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
                    NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
                    NetworkTrainer.NetworkOutputFilePath = "../../../../Results/Networks/";

                    NetworkTrainer.MomentumMultiplier = 0.9;
                    NetworkTrainer.WeightDecayCoeff = 0.0;
                    NetworkTrainer.MaxTrainingEpochs = 1;
                    NetworkTrainer.EpochsBeforeRegularization = 0;
                    NetworkTrainer.MiniBatchSize = 64;
                    NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
                    NetworkTrainer.EvaluateBeforeTraining = true;
                    NetworkTrainer.DropoutFullyConnected = 1.0;
                    NetworkTrainer.DropoutConvolutional = 1.0;
                    NetworkTrainer.Patience = 20;

                    NetworkTrainer.LearningRate = eta[iEta];
                    NetworkTrainer.Train(network, trainingSet, null);
                }

            
            
                // VGG_RELU ____________________________________________________________________________________________________

                for (int iEta = 0; iEta < eta.Length; iEta++)
                {
                    Console.WriteLine("\n\n\n New learning rate = {0}", eta[iEta]);



                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Neural network creation");
                    Console.WriteLine("=========================================\n");

                    // OPTION 1: Create a new network

                    NeuralNetwork network = new NeuralNetwork("EtaTest_VGG_ReLU");

                    network.AddLayer(new InputLayer(1, 32, 32));

                    network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new ConvolutionalLayer(3, 64, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new ConvolutionalLayer(3, 64, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new ConvolutionalLayer(3, 128, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new ConvolutionalLayer(3, 128, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new FullyConnectedLayer(128));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new FullyConnectedLayer(128));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new FullyConnectedLayer(43));
                    network.AddLayer(new SoftMax());

                    NetworkTrainer.TrainingMode = "new";


                    /*****************************************************
                        * (3) Gradient check
                        ****************\*********************************
                    GradientChecker.Check(network, validationSet);





                    /*****************************************************
                        * (4) Train network
                        *****************************************************
            
                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Network training");
                    Console.WriteLine("=========================================\n");

                    // Set output files save paths
                    string trainingSavePath = "../../../../Results/LossError/";
                    NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
                    NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
                    NetworkTrainer.NetworkOutputFilePath = "../../../../Results/Networks/";

                    NetworkTrainer.MomentumMultiplier = 0.9;
                    NetworkTrainer.WeightDecayCoeff = 0.0;
                    NetworkTrainer.MaxTrainingEpochs = 1;
                    NetworkTrainer.EpochsBeforeRegularization = 0;
                    NetworkTrainer.MiniBatchSize = 64;
                    NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
                    NetworkTrainer.EvaluateBeforeTraining = true;
                    NetworkTrainer.DropoutFullyConnected = 1.0;
                    NetworkTrainer.DropoutConvolutional = 1.0;
                    NetworkTrainer.Patience = 20;

                    NetworkTrainer.LearningRate = eta[iEta];
                    NetworkTrainer.Train(network, trainingSet, null);
                }



                // VGG_ELU ____________________________________________________________________________________________________

                for (int iEta = 0; iEta < eta.Length; iEta++)
                {
                    Console.WriteLine("\n\n\n New learning rate = {0}", eta[iEta]);



                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Neural network creation");
                    Console.WriteLine("=========================================\n");

                    // OPTION 1: Create a new network

                    NeuralNetwork network = new NeuralNetwork("EtaTest_VGG_ELU");

                    network.AddLayer(new InputLayer(1, 32, 32));

                    network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new ConvolutionalLayer(3, 64, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new ConvolutionalLayer(3, 64, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new ConvolutionalLayer(3, 128, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new ConvolutionalLayer(3, 128, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new MaxPooling(2, 2));

                    network.AddLayer(new FullyConnectedLayer(128));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new FullyConnectedLayer(128));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new FullyConnectedLayer(43));
                    network.AddLayer(new SoftMax());

                    NetworkTrainer.TrainingMode = "new";


                    /*****************************************************
                        * (3) Gradient check
                        ****************\*********************************
                    GradientChecker.Check(network, validationSet);





                    /*****************************************************
                        * (4) Train network
                        *****************************************************
            
                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Network training");
                    Console.WriteLine("=========================================\n");

                    // Set output files save paths
                    string trainingSavePath = "../../../../Results/LossError/";
                    NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
                    NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
                    NetworkTrainer.NetworkOutputFilePath = "../../../../Results/Networks/";

                    NetworkTrainer.MomentumMultiplier = 0.9;
                    NetworkTrainer.WeightDecayCoeff = 0.0;
                    NetworkTrainer.MaxTrainingEpochs = 1;
                    NetworkTrainer.EpochsBeforeRegularization = 0;
                    NetworkTrainer.MiniBatchSize = 64;
                    NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
                    NetworkTrainer.EvaluateBeforeTraining = true;
                    NetworkTrainer.DropoutFullyConnected = 1.0;
                    NetworkTrainer.DropoutConvolutional = 1.0;
                    NetworkTrainer.Patience = 20;

                    NetworkTrainer.LearningRate = eta[iEta];
                    NetworkTrainer.Train(network, trainingSet, null);
                }



            */
                // RESNET_RELU ____________________________________________________________________________________________________
                /*
                for (int iEta = 0; iEta < eta.Length; iEta++)
                {
                    Console.WriteLine("\n\n\n New learning rate = {0}", eta[iEta]);



                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Neural network creation");
                    Console.WriteLine("=========================================\n");

                    // OPTION 1: Create a new network

                    NeuralNetwork network = new NeuralNetwork("EtaTest_ResNet_ReLU");

                    network.AddLayer(new InputLayer(1, 32, 32));

                    network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new ResidualModule(3, 32, 1, 1, "ReLU"));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new ConvolutionalLayer(3, 64, 2, 1)); // downsampling

                    network.AddLayer(new ResidualModule(3, 64, 1, 1, "ReLU"));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new ConvolutionalLayer(3, 128, 2, 1)); // downsampling

                    network.AddLayer(new ResidualModule(3, 128, 1, 1, "ReLU"));
                    network.AddLayer(new ReLU());

                    network.AddLayer(new AveragePooling());

                    network.AddLayer(new FullyConnectedLayer(43));
                    network.AddLayer(new SoftMax());

                    NetworkTrainer.TrainingMode = "new";


                    /*****************************************************
                        * (3) Gradient check
                        ****************\*********************************
                    GradientChecker.Check(network, validationSet);





                    /*****************************************************
                        * (4) Train network
                        *****************************************************/
            /*
                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Network training");
                    Console.WriteLine("=========================================\n");

                    // Set output files save paths
                    string trainingSavePath = "../../../../Results/LossError/";
                    NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
                    NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
                    NetworkTrainer.NetworkOutputFilePath = "../../../../Results/Networks/";

                    NetworkTrainer.MomentumMultiplier = 0.9;
                    NetworkTrainer.WeightDecayCoeff = 0.0;
                    NetworkTrainer.MaxTrainingEpochs = 1;
                    NetworkTrainer.EpochsBeforeRegularization = 0;
                    NetworkTrainer.MiniBatchSize = 64;
                    NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
                    NetworkTrainer.EvaluateBeforeTraining = true;
                    NetworkTrainer.DropoutFullyConnected = 1.0;
                    NetworkTrainer.DropoutConvolutional = 1.0;
                    NetworkTrainer.Patience = 20;

                    NetworkTrainer.LearningRate = eta[iEta];
                    NetworkTrainer.Train(network, trainingSet, null);
                }

            */

                // RESNET_ELU ____________________________________________________________________________________________________

                for (int iEta = 0; iEta < eta.Length; iEta++)
                {
                    Console.WriteLine("\n\n\n New learning rate = {0}", eta[iEta]);



                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Neural network creation");
                    Console.WriteLine("=========================================\n");

                    // OPTION 1: Create a new network


                    NeuralNetwork network = new NeuralNetwork("EtaTest_ResNet_ReLU");

                    network.AddLayer(new InputLayer(1, 32, 32));

                    network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new ResidualModule(3, 32, 1, 1, "ELU"));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new ConvolutionalLayer(3, 64, 2, 1)); // downsampling

                    network.AddLayer(new ResidualModule(3, 64, 1, 1, "ELU"));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new ConvolutionalLayer(3, 128, 2, 1)); // downsampling

                    network.AddLayer(new ResidualModule(3, 128, 1, 1, "ELU"));
                    network.AddLayer(new ELU(1.0f));

                    network.AddLayer(new AveragePooling());

                    network.AddLayer(new FullyConnectedLayer(43));
                    network.AddLayer(new SoftMax());

                    NetworkTrainer.TrainingMode = "new";

                    /*****************************************************
                        * (3) Gradient check
                        ****************\*********************************
                    GradientChecker.Check(network, validationSet);





                    /*****************************************************
                        * (4) Train network
                        *****************************************************/

            
                    Console.WriteLine("\n=========================================");
                    Console.WriteLine("    Network training");
                    Console.WriteLine("=========================================\n");

                    // Set output files save paths
                    string trainingSavePath = "../../../../Results/LossError/";
                    NetworkTrainer.TrainingEpochSavePath = trainingSavePath + network.Name + "_trainingEpochs.txt";
                    NetworkTrainer.ValidationEpochSavePath = trainingSavePath + network.Name + "_validationEpochs.txt";
                    NetworkTrainer.NetworkOutputFilePath = "../../../../Results/Networks/";

                    NetworkTrainer.MomentumMultiplier = 0.9;
                    NetworkTrainer.WeightDecayCoeff = 0.0;
                    NetworkTrainer.MaxTrainingEpochs = 1;
                    NetworkTrainer.EpochsBeforeRegularization = 0;
                    NetworkTrainer.MiniBatchSize = 64;
                    NetworkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
                    NetworkTrainer.EvaluateBeforeTraining = true;
                    NetworkTrainer.DropoutFullyConnected = 1.0;
                    NetworkTrainer.DropoutConvolutional = 1.0;
                    NetworkTrainer.Patience = 20;

                    NetworkTrainer.LearningRate = eta[iEta];
                    NetworkTrainer.Train(network, trainingSet, null);
                }




            /*****************************************************
             * (5) Test network
             *****************************************************
            Console.WriteLine("\nFINAL EVALUATION:");


            // Load best network from file
            NeuralNetwork bestNetwork = Utils.LoadNetworkFromFile("../../../../Results/Networks/", network.Name);
            bestNetwork.Set("MiniBatchSize", 64); // this SHOULDN'T matter!
            bestNetwork.InitializeParameters("load");
            bestNetwork.Set("Inference", true);

            double loss;
            double error;

            // Pre-inference pass: Computes cumulative averages in BatchNorm layers (needed for evaluation)
            //bestNetwork.Set("PreInference", true);
            //networkEvaluator.PreEvaluateNetwork(bestNetwork, testSet);

            

            //networkEvaluator.EvaluateNetwork(bestNetwork, trainingSet, out loss, out error);
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
            if (bestNetwork.Layers[1].Type == "Convolutional")
                Utils.SaveFilters(bestNetwork, "../../../../Results/Filters/" + network.Name + "_filters.txt");
            
            /*****************************************************/
        }
    }
}
