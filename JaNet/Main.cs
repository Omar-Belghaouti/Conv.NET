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

            network.AddLayer(new InputLayer(1, 28, 28));

            network.AddLayer(new ConvolutionalLayer(5, 8, 1, 0));
            network.AddLayer(new ReLU());

            network.AddLayer(new ConvolutionalLayer(5, 8, 1, 0));
            network.AddLayer(new ReLU());


            network.AddLayer(new FullyConnectedLayer(16));
            //network.AddLayer(new Tanh(0.5));
            network.AddLayer(new ReLU());

            network.AddLayer(new FullyConnectedLayer(10));
            network.AddLayer(new SoftMax());



            /*****************************************************
             * (2) Load data
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");

            #region Paths to datasets

            // Original MNIST training set
            string MNISTtrainingData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainImages.dat";
            string MNISTtrainingLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainLabels.dat";

            // Original MNIST test set
            string MNISTtestData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestImages.dat";
            string MNISTtestLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestLabels.dat";

            // Reduced MNIST dataset (1000 data points, 100 per digit)
            string MNISTreducedData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistImagesSubset.dat";
            string MNISTreducedLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistLabelsSubset.dat";

            // GTSRB training set (WARNING: counterfait!)
            string GTSRBtrainingData = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/02_training_images.dat";
            string GTSRBtrainingLabels = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/training_labels_full.dat";

            // GTSRB test set (RGB)
            string GTSRBtestDataRGB = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/03_test_images.dat";
            string GTSRBtestLabelsRGB = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/test_labels_full.dat";

            // GTSRB test set (grayscale)
            string GTSRBtestDataGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/02_test_images.dat";
            string GTSRBtestLabelsGS = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/test_labels_full.dat";

            #endregion

            Console.WriteLine("Importing training set...");

            DataSet trainingSet = new DataSet(10);

            trainingSet.ReadData(MNISTreducedData);
            trainingSet.ReadLabels(MNISTreducedLabels);

            //Console.WriteLine("Importing validation set...");

            //DataSet validationSet = new DataSet(10);

            //validationSet.ReadData(MNISTreducedData);
            //validationSet.ReadLabels(MNISTreducedLabels);

            /*****************************************************
             * (3) Train network
             ****************\************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");



            NetworkTrainer networkTrainer = new NetworkTrainer(network, trainingSet, null);

            networkTrainer.LearningRate = 0.0005;
            networkTrainer.MomentumMultiplier = 0.9;
            networkTrainer.MaxTrainingEpochs = 1000;
            networkTrainer.MiniBatchSize = 8; // property includes buffer increase
            networkTrainer.ErrorTolerance = 0.0;
            networkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
            networkTrainer.EvaluateBeforeTraining = true;
            networkTrainer.EarlyStopping = false;
            
            
            networkTrainer.Train();
            

            

            /*****************************************************
             * (4) Test network
             ****************************************************/

            /*
            NetworkEvaluator networkEvaluator = new NetworkEvaluator();

            double loss;
            double error;
            networkEvaluator.EvaluateNetwork(network, trainingSet, out loss, out error);
            Console.WriteLine("Final evaluation\n\tLoss = {0}\n\tError = {1}", loss, error);
#if GRADIENT_CHECK
            GradientChecker.Check(network, reducedMNIST);
#endif
            */
        }
    }
}
