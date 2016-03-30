using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;
using System.Diagnostics;

namespace JaNet
{
    class NeuralNetwork
    {
        #region NeuralNetwork class fields (private)

        private List<Layer> layers;
        private int nLayers = 0;

        #endregion


        #region NeuralNetwork class properties (public)

        public List<Layer> Layers
        {
            get { return layers; }
        }
        public int NumberOfLayers
        {
            get { return nLayers; }
        }

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// NeuralNetwork class constructor.
        /// </summary>
        public NeuralNetwork()
        {
            //Console.WriteLine("--- New empty network created ---\n");
            this.layers = new List<Layer>(); // empty list of layers
        }

        /// <summary>
        /// Add layer to NeuralNetwork object.
        /// </summary>
        /// <param name="layer"></param>
        public void AddLayer(Layer layer)
        {
            if (this.layers.Any()) // if layer list is not empty
                this.layers.Last().NextLayer = layer; // set this layer as layer field of previous one

            this.layers.Add(layer);
            this.nLayers++;
        }

        /// <summary>
        /// Setup network: given input dim and each layer's parameters, automatically set dimensions of I/O 3D arrays and initialize weights and biases.
        /// </summary>
        /// <param name="inputDimensions"></param>
        /// <param name="nOutputClasses"></param>
        public void Setup(int inputWidth, int inputHeigth, int inputDepth, int nOutputClasses)
        {
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    Network setup and initialization");
            Console.WriteLine("=========================================\n");

            Console.WriteLine("Setting up layer 0 (input layer): " + layers[0].Type);
            layers[0].SetAsFirstLayer(inputWidth, inputHeigth, inputDepth); 
            layers[0].InitializeParameters();

            for (int i = 1; i < layers.Count; i++ ) // all other layers
            {
                Console.WriteLine("Setting up layer " + i.ToString() + ": " + layers[i].Type);
                layers[i].ConnectTo(layers[i - 1]);
                layers[i].InitializeParameters();
                
            }
        }

        #endregion


        #region Other methods
        [Obsolete("Now implemented in class NetworkTrainer")]
        public int RunForwardOne(float[] inputImage, out float[] outputClassScores)
        {
            int errorCode = 0;

            layers[0].Input.SetHost(inputImage);
            for (int iLayer = 0; iLayer < nLayers - 1; iLayer++) // all layers but last
            {
                layers[iLayer].FeedForward();
                layers[iLayer + 1].Input = layers[iLayer].Output; // set input of next layer equals to output of this layer
            }
            layers[nLayers - 1].FeedForward(); // run output (softmax) layer
            outputClassScores = (float[])layers[nLayers - 1].Output.GetHost();

            return errorCode;
        }

        #endregion


    }

    
}
