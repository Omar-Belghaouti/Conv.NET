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
            NetworkTrainer.LearningRate = 0.0001;
            NetworkTrainer.MomentumMultiplier = 0.9;
            NetworkTrainer.MaxTrainingEpochs = 100;
            NetworkTrainer.MiniBatchSize = 90; // use multiples of 30
            NetworkTrainer.ErrorTolerance = 0.001;


            /*****************************************************
             * (1) Instantiate a neural network and add layers
             ****************************************************/
            NeuralNetwork net = new NeuralNetwork();
            // neuralNet.AddLayer(new ConvolutionalLayer(7,40));
            net.AddLayer(new FullyConnectedLayer(100));
            net.AddLayer(new FullyConnectedLayer(100));
            //net.AddLayer(new SoftMaxLayer(43));



            /*****************************************************
             * (2) Load data
             ****************************************************/

            // data will be preprocessed and split into training/validation sets with MATLAB
            DataSet trainingSet = new DataSet();
            DataSet validationSet = new DataSet();

            net.Setup(32, 32, 3, 43); // input img: 32x32x3, output classes: 43

            /*
            for (int i = 0; i < net.layers.Count; i++)
            {
                Console.WriteLine("\nLayer {0} is a {1}", i, net.layers[i].GetType());
                Console.WriteLine("\tInput dimensions: {0}, {1}, {2}",
                    net.layers[i].InputWidth, net.layers[i].InputHeight, net.layers[i].InputDepth);
                Console.WriteLine("\tOutput dimensions: {0}, {1}, {2}",
                    net.layers[i].OutputWidth, net.layers[i].OutputHeight, net.layers[i].OutputDepth);

            }
             */

            /*****************************************************
             * (3) Train network
             ****************************************************/
            errorCode = NetworkTrainer.Run(net, trainingSet, validationSet);



            /*****************************************************
             * (4) Test network
             ****************************************************/
           


        }
    }
}
