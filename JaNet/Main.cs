//#define OPENCL_ENABLED

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class trafficNetCLProgram
    {
        
        static void Main(string[] args)
        {

            /*****************************************************
             * (0a) Set hyperparameters
             ****************************************************/
            NetworkTrainer.LearningRate = 0.001;
            NetworkTrainer.MomentumMultiplier = 0.9;
            NetworkTrainer.MaxTrainingEpochs = 20;
            NetworkTrainer.MiniBatchSize = 1; // not correctly implemented yet!! // for GTSRB can use any multiple of 2, 3, 5
            NetworkTrainer.ErrorTolerance = 0.001;
            NetworkTrainer.ConsoleOutputLag = 1000; // 1 = print every epoch, N = print every N epochs
            //double tanhBeta = 0.5;
            

            /*****************************************************
             * (0b) Setup OpenCL
             ****************************************************/
#if OPENCL_ENABLED
            CL.Setup();
            const string kernelsPath = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/JaNet/Kernels";
            CL.LoadKernels(kernelsPath);
#endif

            

            

            

            /*****************************************************
             * (1) Instantiate a neural network and add layers
             * 
             * OPTIONS:
             * ConvolutionalLayer(filterSize, numberOfFilters, strideLength, zeroPadding) // zero padding not implemented yet
             * FullyConnectedLayer(numberOfUnits)
             * Tanh(beta)
             * ReLU()
             * SoftMax()
             ****************************************************/

            NeuralNetwork network = new NeuralNetwork();
            //network.AddLayer(new ConvolutionalLayer(3, 16, 1, 0));
            //network.AddLayer(new ConvolutionalLayer(3, 16, 1, 0));
            //network.AddLayer(new ReLU());

            //network.AddLayer(new Tanh(tanhBeta));
            network.AddLayer(new FullyConnectedLayer(32));
            network.AddLayer(new ReLU());
            //network.AddLayer(new FullyConnectedLayer(16));
            //network.AddLayer(new ReLU());
            //network.AddLayer(new Tanh(tanhBeta));
            //network.AddLayer(new FullyConnectedLayer(128));
            //network.AddLayer(new ReLU());
            //network.AddLayer(new Tanh(tanhBeta));
            network.AddLayer(new FullyConnectedLayer(10));
            network.AddLayer(new SoftMax());


            /*****************************************************
             * (2) Load data
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");

            // data will be preprocessed and split into training/validation sets with MATLAB
            
            // Simple, 2-dimensional data set, for initial testing
            //DataSet dataSet = new DataSet(2, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/JaNet/Data/train_data_centered_scaled.txt");





            /*
            // Reduced MNIST dataset (1000 data points, 100 per digit)
            DataSet dataSet = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistImagesSubset.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistLabelsSubset.dat");
            */


            // Original MNIST dataset
            
            //Console.WriteLine("Importing training data...");
            /*
            DataSet dataSet = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainImages.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainLabels.dat");
            */

            
            //Console.WriteLine("Importing test data...");
            
            DataSet dataSet = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestImages.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestLabels.dat");
            

            //network.Setup(2, 1, 1, 2); // toy dataset
            network.Setup(28, 28, 1, 10); // MNIST

#if OPENCL_ENABLED
            NetworkTrainer.SetupCL(dataSet);
            NetworkEvaluator.SetupCL(dataSet, NetworkTrainer.MiniBatchSize);
#endif

            /*****************************************************
             * (3) Train network
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");


            //NetworkTrainer.TrainSimpleTest(network, dataSet);

            
            //network.Layers[0].DisplayParameters();
            //network.Layers[2].DisplayParameters();
            //network.Layers[4].DisplayParameters();
            


            NetworkTrainer.TrainMNIST(network, dataSet);

            /*****************************************************
             * (4) Test network
             ****************************************************/
           



            }
    }
}
