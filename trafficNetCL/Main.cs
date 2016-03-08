using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace trafficNetCL
{
    class trafficNetCLProgram
    {

        static void Main(string[] args)
        {

            // (0) Set hyperparameters

            double learnRate = 0.05; 
            double momentum = 0.01; 
            int maxEpochs = 1000; 
            Console.WriteLine("\nSetting learning rate = " + learnRate.ToString("F2")); 
            Console.WriteLine("Setting momentum = " + momentum.ToString("F2")); 
            Console.WriteLine("Setting max epochs = " + maxEpochs + "\n");

            // (1) Create a neural network
            NeuralNetwork neuralNet = new NeuralNetwork();
            neuralNet.AddLayer(new Layer(1,1,1,2,2,2));
            neuralNet.AddLayer(new ConvolutionalLayer(1,2,3,4,5,6));
            // etc...
            /* neuralNet.AddLayer(new ReLU( <layer parameters> );
             * neuralNet.AddLayer(new PoolingLayer( <layer parameters> );
             * neuralNet.AddLayer(new ConvolutionalLayer( <layer parameters> );
             * neuralNet.AddLayer(new ReLU( <layer parameters> );
             * neuralNet.AddLayer(new PoolingLayer( <layer parameters> );
             * neuralNet.AddLayer(new FullyConnectedLayer( <layer parameters> );
             * neuralNet.AddLayer(new FullyConnectedLayer( <layer parameters> );
             * neuralNet.AddLayer(new OutputLayer( <layer parameters> );
             */

            // (2) Load data

            // (3) Train network

            // (4) Test network


        }
    }
}
