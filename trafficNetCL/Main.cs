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

            /*****************************************************
             * (0) Set hyperparameters
             ****************************************************/

            double learnRate = 0.05; 
            double momentum = 0.01; 
            int maxEpochs = 100; 
            Console.WriteLine("\nLearning rate = " + learnRate.ToString("F2")); 
            Console.WriteLine("Momentum = " + momentum.ToString("F2")); 
            Console.WriteLine("Max epochs = " + maxEpochs + "\n");


            

            /*****************************************************
             * (1) Instantiate a neural network and add layers
             ****************************************************/
            NeuralNetwork neuralNet = new NeuralNetwork();
            // neuralNet.AddLayer(new ConvolutionalLayer(7,40));
            neuralNet.AddLayer(new FullyConnectedLayer(100) );
            neuralNet.AddLayer(new FullyConnectedLayer(200) );
            neuralNet.AddLayer(new FullyConnectedLayer(300) );
            neuralNet.AddLayer(new SoftMaxLayer(43) );

            Console.WriteLine("\nBefore setup:");

            for (int i = 0; i < neuralNet.Layers.Count; i++)
            {
                Console.WriteLine("\nLayer {0} is a {1}", i, neuralNet.Layers[i].GetType());
                Console.WriteLine("\tInput dimensions: {0}, {1}, {2}",
                    neuralNet.Layers[i].InputWidth, neuralNet.Layers[i].InputHeight, neuralNet.Layers[i].InputDepth);
                Console.WriteLine("\tOutput dimensions: {0}, {1}, {2}",
                    neuralNet.Layers[i].OutputWidth, neuralNet.Layers[i].OutputHeight, neuralNet.Layers[i].OutputDepth);

            }

            // EXPERIMENTS HERE:
            /*
            Console.WriteLine("Layer 1 points to {0}",neuralNet.Layers[0].NextLayer);
            Console.Write("Layer 2 points to {0}", neuralNet.Layers[1].NextLayer);
            if (neuralNet.Layers[1].NextLayer == null)
                Console.Write("NULL\n");
            */

            /*****************************************************
             * (2) Load data
             ****************************************************/

            // data will be preprocessed and split into training/validation sets with MATLAB

            int[] inputDimensions = new int[] { 32, 32, 3 };
            int nOutputClasses = 43;
            neuralNet.Setup(inputDimensions, nOutputClasses); // input img: 32x32x3, output classes: 43


            Console.WriteLine("\nAFTER SETUP\n");


            for (int i = 0; i < neuralNet.Layers.Count; i++)
            {
                Console.WriteLine("\nLayer {0} is a {1}", i, neuralNet.Layers[i].GetType());
                Console.WriteLine("\tInput dimensions: {0}, {1}, {2}", 
                    neuralNet.Layers[i].InputWidth, neuralNet.Layers[i].InputHeight, neuralNet.Layers[i].InputDepth);
                Console.WriteLine("\tOutput dimensions: {0}, {1}, {2}",
                    neuralNet.Layers[i].OutputWidth, neuralNet.Layers[i].OutputHeight, neuralNet.Layers[i].OutputDepth);

            }

            /*****************************************************
             * (3) Train network
             ****************************************************/




            /*****************************************************
             * (4) Test network
             ****************************************************/
           


        }
    }
}
