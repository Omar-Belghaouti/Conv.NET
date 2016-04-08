﻿//#define OPENCL_ENABLED

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
             * ConvolutionalLayer(filterSize, numberOfFilters, strideLength, zeroPadding) // zero padding not implemented yet
             * FullyConnectedLayer(numberOfUnits)
             * Tanh(beta)
             * ReLU()
             * SoftMax()
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Neural network creation");
            Console.WriteLine("=========================================\n");

            NeuralNetwork network = new NeuralNetwork();
            network.AddLayer(new InputLayer(1, 1, 32, 32));

            network.AddLayer(new FullyConnectedLayer(64));
            network.AddLayer(new ReLU());

            network.AddLayer(new FullyConnectedLayer(43));
            network.AddLayer(new SoftMax());
            

            /*****************************************************
             * (2) Load data
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");



            // Original MNIST training set
            //DataSet trainingSetMNIST = new DataSet(10, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainImages.dat", "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainLabels.dat");

            // Original MNIST test set
            //DataSet testSetMNIST = new DataSet(10, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestImages.dat", "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestLabels.dat");

            // Reduced MNIST dataset (1000 data points, 100 per digit)
            // DataSet reducedMNIST = new DataSet(10, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistImagesSubset.dat", "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistLabelsSubset.dat");

            // GTSRB training set
            DataSet trainingGTSRB = new DataSet(43, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/02_training_images.dat", "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/training_labels_full.dat");

            // GTSRB test set
            DataSet testGTSRB = new DataSet(43, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/02_test_images.dat", "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/test_labels_full.dat");


            /*****************************************************
             * (3) Train network
             ****************\************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");

            NetworkTrainer networkTrainer = new NetworkTrainer(network, trainingGTSRB, testGTSRB);

            networkTrainer.LearningRate = 0.0005;
            networkTrainer.MomentumMultiplier = 0.9;
            networkTrainer.MaxTrainingEpochs = 1000;
            networkTrainer.MiniBatchSize = 1; // not correctly implemented yet!! // for GTSRB can use any multiple of 2, 3, 5
            networkTrainer.ErrorTolerance = 0.0;
            networkTrainer.ConsoleOutputLag = 1; // 1 = print every epoch, N = print every N epochs
            networkTrainer.EvaluateBeforeTraining = true;
            networkTrainer.EarlyStopping = true;
            //double tanhBeta = 0.5;

            networkTrainer.Train();


            /*****************************************************
             * (4) Test network
             ****************************************************/


        }
    }
}
