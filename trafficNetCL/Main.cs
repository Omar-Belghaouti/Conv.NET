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
            NetworkTrainer.LearningRate = 0.01;
            NetworkTrainer.MomentumMultiplier = 0.9;
            NetworkTrainer.MaxTrainingEpochs = 100;
            NetworkTrainer.MiniBatchSize = 1; // not correctly implemented yet!! // for GTSRB can use any multiple of 2, 3, 5
            NetworkTrainer.ErrorTolerance = 0.001;
            NetworkTrainer.ConsoleOutputLag = 1;
            //double tanhBeta = 0.5;
            

            /*****************************************************
             * (0b) Setup OpenCL
             ****************************************************/
            CL.Setup();
            const string kernelsPath = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/JaNet/Kernels";
            CL.LoadKernels(kernelsPath);

            

            

            

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
            network.AddLayer(new FullyConnectedLayer(37));
            network.AddLayer(new ReLU());
            //network.AddLayer(new Tanh(tanhBeta));
            //network.AddLayer(new FullyConnectedLayer(100));
            //network.AddLayer(new ReLU());
            //network.AddLayer(new Tanh(tanhBeta));
            network.AddLayer(new FullyConnectedLayer(10));
            network.AddLayer(new SoftMax());


            /*****************************************************
             * (2) Load data
             ****************************************************/

            // data will be preprocessed and split into training/validation sets with MATLAB
            
            // Simple, 2-dimensional data set, for initial testing
            //DataSet trainingSet = new DataSet(2, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/TrafficNetCL/Data/train_data_centered_scaled.txt");

            // MNIST dataset reduced to 1000 data points, 
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");
            DataSet reducedMNIST = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistImagesSubset.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistLabelsSubset.dat");

            /*
            Console.WriteLine("Importing training data...");
            DataSet trainingSet = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainImages.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainLabels.dat");
            

            Console.WriteLine("Importing test data...");
            DataSet testSet = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestImages.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestLabels.dat");
             */

            network.Setup(28, 28, 1, 10);



            /*****************************************************
             * (3) Train network
             ****************************************************/
            //errorCode = NetworkTrainer.Train(net, trainingSet, validationSet);

            /*
            double errorTraining;
            int finalEpoch;
            errorCode = NetworkTrainer.TrainSimpleTest(network, reducedMNIST, out errorTraining, out finalEpoch);

            
            network.Layers[0].DisplayParameters();
            network.Layers[2].DisplayParameters();
            network.Layers[4].DisplayParameters();
            */


            double errorMNIST;
            errorMNIST = NetworkTrainer.TrainMNIST(network, reducedMNIST);


            /*****************************************************
             * (4) Test network
             ****************************************************/
           



            }
    }
}
