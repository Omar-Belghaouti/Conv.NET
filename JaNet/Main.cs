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
            /*****************************************************
             * Training hyperparameters
             ****************************************************/
            NetworkTrainer trainer = new NetworkTrainer();

            trainer.LearningRate = 0.0005;
            trainer.MomentumMultiplier = 0.9;
            trainer.MaxTrainingEpochs = 1000;
            trainer.MiniBatchSize = 1; // not correctly implemented yet!! // for GTSRB can use any multiple of 2, 3, 5
            trainer.ErrorTolerance = 0.0;
            trainer.ConsoleOutputLag = 5; // 1 = print every epoch, N = print every N epochs
            trainer.EvaluateBeforeTraining = true;
            //double tanhBeta = 0.5;

            
            

            /*****************************************************
             * (0b) Setup OpenCL
             ****************************************************/
#if OPENCL_ENABLED
            const string kernelsPath = "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/JaNet/Kernels";
            CL.Setup(kernelsPath);
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

            network.AddLayer(new InputLayer(1, 1, 28, 28));

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

            trainer.Network = network;


            /*****************************************************
             * (2) Load data
             ****************************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Importing data");
            Console.WriteLine("=========================================\n");

            // data will be preprocessed and split into training/validation sets with MATLAB
            
            // Simple, 2-dimensional data set, for initial testing
            //DataSet dataSet = new DataSet(2, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/JaNet/Data/train_data_centered_scaled.txt");





            
            
            


            // Original MNIST dataset
            
            //Console.WriteLine("Importing training data...");
            /*
            DataSet trainingSetMNIST = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainImages.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTrainLabels.dat");
            */

            
            //Console.WriteLine("Importing test data...");
            
            /*
            DataSet testSetMNIST = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestImages.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistTestLabels.dat");
            */

            
            // Reduced MNIST dataset (1000 data points, 100 per digit)
            DataSet reducedMNIST = new DataSet(10,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistImagesSubset.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/MNIST/mnistLabelsSubset.dat");
            

            // GTSRB training set (full)
            /*
            DataSet trainingGTSRB = new DataSet(43,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/02_training_images.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/training_labels_full.dat");

            DataSet testGTSRB = new DataSet(43,
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/02_test_images.dat",
                "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/GTSRB/Preprocessed/test_labels_full.dat");
            */
            
            //network.Setup(2, 1, 1, 2); // toy dataset
            //network.Setup(28, 28, 1, 10); // MNIST
            //network.Setup(32, 32, 1, 43); // GTSRB


            trainer.TrainingSet = reducedMNIST;
            //trainer.ValidationSet = testGTSRB;

            
            /*****************************************************
             * (3) Train network
             ****************\************************************/
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network training");
            Console.WriteLine("=========================================\n");


            //NetworkTrainer.TrainSimpleTest(network, dataSet);

            
            //network.Layers[0].DisplayParameters();
            //network.Layers[2].DisplayParameters();
            //network.Layers[4].DisplayParameters();


            trainer.Train();


            /*****************************************************
             * (4) Test network
             ****************************************************/
           



            }
    }
}
