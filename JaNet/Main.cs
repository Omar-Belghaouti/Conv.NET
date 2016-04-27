//#define OPENCL_ENABLED

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

#if OPENCL_ENABLED

            /*****************************************************
             * (0) Setup OpenCL
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    OpenCL setup");
            Console.WriteLine("=========================================\n");

            OpenCLSpace.SetupSpace();
            OpenCLSpace.KernelsPath = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/JaNet/Kernels";
            OpenCLSpace.LoadKernels();  
#endif


            /*****************************************************
             * (1) Instantiate a neural network and add layers
             * 
             * OPTIONS:
             * ConvolutionalLayer(filterSize, numberOfFilters, strideLength, zeroPadding)
             * FullyConnectedLayer(numberOfUnits)
             * Tanh(beta)
             * ReLU()
             * SoftMax()
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Neural network creation");
            Console.WriteLine("=========================================\n");

            NeuralNetwork network = new NeuralNetwork();

            network.AddLayer(new InputLayer(1, 32, 32));
        
            network.AddLayer(new ConvolutionalLayer(3, 16, 1, 1));
            //network.AddLayer(new ReLU());
            network.AddLayer(new ELU(1.0f));
            //network.AddLayer(new ConvolutionalLayer(3, 8, 1, 1));
            //network.AddLayer(new ReLU());
            //network.AddLayer(new ELU(1.0f));
            //network.AddLayer(new PoolingLayer("max", 2, 2));

            network.AddLayer(new ConvolutionalLayer(3, 16, 1, 1));
            network.AddLayer(new ELU(1.0f));
            //network.AddLayer(new ReLU());
            //network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
            //network.AddLayer(new ELU(1.0f));
            network.AddLayer(new PoolingLayer("max", 2, 2));

            network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
            network.AddLayer(new ELU(1.0f));
            network.AddLayer(new ConvolutionalLayer(3, 32, 1, 1));
            network.AddLayer(new ELU(1.0f));
            //network.AddLayer(new ConvolutionalLayer(3, 64, 1, 1));
            //network.AddLayer(new ELU(1.0f));
            //network.AddLayer(new ReLU());
            network.AddLayer(new PoolingLayer("max", 2, 2));

            //network.AddLayer(new FullyConnectedLayer(100));
            //network.AddLayer(new ReLU());

            network.AddLayer(new FullyConnectedLayer(128));
            network.AddLayer(new ELU(1.0f));

            //network.AddLayer(new FullyConnectedLayer(128));
            //network.AddLayer(new ELU(1.0f));

            network.AddLayer(new FullyConnectedLayer(43));
            network.AddLayer(new SoftMax());



            /*****************************************************
             * (2) Load data
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");

            #region Paths to datasets

            // Original MNIST training set
            //string MNISTtrainingData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainImages.dat";
            //string MNISTtrainingLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainLabels.dat";

            // Original MNIST test set
            //string MNISTtestData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestImages.dat";
            //string MNISTtestLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestLabels.dat";

            // Reduced MNIST dataset (1000 data points, 100 per digit)
            //string MNISTreducedData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistImagesSubset.dat";
            //string MNISTreducedLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistLabelsSubset.dat";

            // GTSRB training set (grayscale)
            string GTSRBtrainingDataGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/08_training_images.dat";
            string GTSRBtrainingLabelsGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/08_training_classes.dat";

            // GTSRB validation set (grayscale)
            string GTSRBvalidationDataGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/08_validation_images.dat";
            string GTSRBvalidationLabelsGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/08_validation_classes.dat";

            // GTSRB test set (grayscale)
            string GTSRBtestDataGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/08_test_images.dat";
            string GTSRBtestLabelsGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/test_labels_full.dat";

            // GTSRB test set (RGB)
            //string GTSRBtestDataRGB = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/03_test_images.dat";
            //string GTSRBtestLabelsRGB = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/test_labels_full.dat";

            

            #endregion

            // Toy MNIST dataset
            /*
            Console.WriteLine("Importing toy dataset...");
            DataSet toySet = new DataSet(10);
            toySet.ReadData(MNISTreducedData);
            toySet.ReadLabels(MNISTreducedLabels);
            */

            
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
             * (3) Train network
             ****************\************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");



            NetworkTrainer networkTrainer = new NetworkTrainer();

            networkTrainer.LearningRate = 0.05;
            networkTrainer.MomentumMultiplier = 0.9;
            networkTrainer.WeightDecayCoeff = 0.0001;
            networkTrainer.MaxTrainingEpochs = 100;
            networkTrainer.MiniBatchSize = 128;
            networkTrainer.ErrorTolerance = 0.0;
            networkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
            networkTrainer.EvaluateBeforeTraining = false;
            networkTrainer.EarlyStopping = false;
            networkTrainer.DropoutFullyConnected = 0.6;
            networkTrainer.DropoutConvolutional = 0.6;
            networkTrainer.EpochsBeforeDropout = -1;


            networkTrainer.Train(ref network, trainingSet, validationSet);
            

            

            /*****************************************************
             * (4) Test network
             ****************************************************/
            
            NetworkEvaluator networkEvaluator = new NetworkEvaluator();
            double loss;
            double error;
            Console.WriteLine("\nFinal evaluation:");

            networkEvaluator.EvaluateNetwork(network, validationSet, out loss, out error);
            Console.WriteLine("\nValidation set:\n\tLoss = {0}\n\tError = {1}", loss, error);

            networkEvaluator.EvaluateNetwork(network, testSet, out loss, out error);
            Console.WriteLine("\nTest set:\n\tLoss = {0}\n\tError = {1}", loss, error);
            
#if GRADIENT_CHECK
            GradientChecker.Check(network, reducedMNIST);
#endif
            
        }
    }
}
