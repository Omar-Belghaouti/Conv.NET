using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class trafficNetCLProgram
    {
        static int errorCode = 0;

        static void Main(string[] args)
        {

            /*****************************************************
             * (0) Set hyperparameters
             ****************************************************/
            NetworkTrainer.LearningRate = 0.01;
            NetworkTrainer.MomentumMultiplier = 0.9;  // not implemented yet
            NetworkTrainer.MaxTrainingEpochs = 10000;
            NetworkTrainer.MiniBatchSize = 3; // for GTSRB can use any multiple of 2, 3, 5
            NetworkTrainer.ErrorTolerance = 0.001;
            NetworkTrainer.ConsoleOutputLag = 100;
            double tanhBeta = 0.5;


            /*****************************************************
             * (1) Instantiate a neural network and add layers
             ****************************************************/
            NeuralNetwork network = new NeuralNetwork();
            // neuralNet.AddLayer(new ConvolutionalLayer(7,40));
            //net.AddLayer(new FullyConnectedLayer(100));
            network.AddLayer(new FullyConnectedLayer(10));
            network.AddLayer(new ReLU());
            network.AddLayer(new FullyConnectedLayer(10));
            network.AddLayer(new ReLU());
            network.AddLayer(new FullyConnectedLayer(2));
            network.AddLayer(new Tanh(tanhBeta));
            //net.AddLayer(new FullyConnectedLayer(10));
            //net.AddLayer(new SoftMaxLayer(43));


            /*****************************************************
             * (2) Load data
             ****************************************************/

            // data will be preprocessed and split into training/validation sets with MATLAB
            DataSet trainingSet = new DataSet(2, "C:/Users/jacopo/Dropbox/Chalmers/MSc thesis/TrafficNetCL/Data/train_data.txt");
            //DataSet validationSet = new DataSet();

            /*
            for (int iPoint = 0; iPoint < trainingSet.Size; iPoint++)
            {
                Console.WriteLine("Data point number {3} is: {0}, {1} and its class is {2}",
                    trainingSet.GetDataPoint(iPoint)[0], trainingSet.GetDataPoint(iPoint)[1], trainingSet.GetLabel(iPoint), iPoint);
            }
             * */
            


            int[] inputDimensions = new int[] {2, 1, 1};
            int outputDimension = 1;
            network.Setup(inputDimensions, outputDimension);



            /*****************************************************
             * (3) Train network
             ****************************************************/
            //errorCode = NetworkTrainer.Train(net, trainingSet, validationSet);
            
            double errorTraining;
            int finalEpoch;
            errorCode = NetworkTrainer.TrainSimpleTest(network, trainingSet, out errorTraining, out finalEpoch);

            network.Layers[0].DisplayParameters();

            network.Layers[2].DisplayParameters();

            //network.Layers[4].DisplayParameters();

            
            
            
            /*****************************************************
             * (4) Test network
             ****************************************************/
           







            /*****************************************************/
            // GENERAL TO-DO LIST:

            // Using DeltaInput is cumbersome and confusing.
            // Add DeltaOutput alongside DeltaInput and use that for parameters update instead.

            // There is something very weird with the weights update: works if the gradient is ADDED instead of SUBTRACTED. Why?!?
        }
    }
}
