using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;
using System.Diagnostics;

namespace TrafficNetCL
{
    class NeuralNetwork
    {
        // Fields
        List<Layer> layers;
        int nLayers = 0;

        // Properties
        public List<Layer> Layers
        {
            get { return layers; }
        }
        public int Depth
        {
            get { return nLayers; }
        }

        /// <summary>
        /// NeuralNetwork class constructor.
        /// </summary>
        public NeuralNetwork()
        {
            Console.WriteLine("--- New empty network created ---");
            this.layers = new List<Layer>(); // empty list of layers
        }

        /// <summary>
        /// Add layer to NeuralNetwork object.
        /// </summary>
        /// <param name="layer"></param>
        public void AddLayer(Layer layer)
        {
            layers.Add(layer);
            nLayers++;
        }

        /// <summary>
        /// Setup network: given input dim and each layer's parameters, automatically set dimensions of I/O 3D arrays and initialize weights and biases.
        /// </summary>
        /// <param name="inputDimensions"></param>
        /// <param name="nOutputClasses"></param>
        public void Setup(int inputImgWidth, int inputImgHeight, int inputImgDepth, int nOutputClasses)
        {
            Console.WriteLine("--- Network setup and initialization started ---");

            Console.WriteLine("Setting up layer 0 (input layer)...");
            layers[0].Setup(inputImgWidth, inputImgHeight, inputImgDepth); // should AUTOMATICALLY setup both input AND output AND initialize weights and biases

            Console.WriteLine("Is input of layer 0 instantiated OUTSIDE? {0}", layers[0].Input != null);
            //Console.WriteLine("Does output of layer 0 exist? {0}", layers[0].Output != null);

            for (int i = 1; i < layers.Count; i++ ) // all other layers
            {
                Console.WriteLine("Setting up layer {0}...", i);
                //Console.WriteLine("Does output of layer {0} exist? {1}", i-1, layers[i - 1].Output != null);
                //Console.WriteLine("Does input of layer {0} exist? {1}", i, layers[i].Input != null);
                layers[i].Input.ConnectTo(layers[i - 1].Output);
                layers[i].Setup();
            }

            Console.WriteLine("--- Network setup and initialization complete ---");
        }

        public int RunForward(float[, ,] inputImage, out float[] outputClassScores)
        {
            int errorCode = 0;

            layers[0].Input.Set(inputImage);
            for (int iLayer = 0; iLayer < nLayers - 1; iLayer++) // all layers but last
            {
                layers[iLayer].ForwardOne();
                layers[iLayer + 1].Input = layers[iLayer].Output; // set input of next layer equals to output of this layer
            }
            layers[nLayers-1].ForwardOne(); // run output (softmax) layer
            outputClassScores = (float[])layers[nLayers - 1].Output.Get();

            return errorCode;
        }





        
        



        

        
		
    }

    
}
