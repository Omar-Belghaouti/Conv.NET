using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Data;
using System.Diagnostics;
using OpenCL.Net;

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



#if OPENCL_ENABLED
        public void ForwardPass(Mem inputDataBatch, int inputBufferBytesSize)
        {
            //TODO: generalise to miniBatchSize > 1

            
            // Copy data point in input buffer of the first layer
            Cl.EnqueueCopyBuffer(   CL.Queue,
                                    inputDataBatch,                      // source
                                    layers[0].Input.ActivationsGPU, // destination
                                    (IntPtr)null,
                                    (IntPtr)null,
                                    (IntPtr)inputBufferBytesSize,
                                    0,
                                    null,
                                    out CL.Event);
            CL.CheckErr(CL.Error, "NeuralNetwork.ForwardPass(): Cl.EnqueueCopyBuffer");

            // Run network forward
            for (int l = 0; l < nLayers; l++)
            {


                /* ------------------------- DEBUGGING ---------------------------------------------

                // Display input layer-by-layer

                float[] layerInput = new float[layers[l].Input.NumberOfUnits];
                CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                layers[l].Input.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(layers[l].Input.NumberOfUnits * sizeof(float)),
                                                layerInput,  // destination
                                                0,
                                                null,
                                                out CL.Event);
                CL.CheckErr(CL.Error, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");

                Console.WriteLine("\nLayer {0} ({1}) input activations:",l , layers[l].Type);
                for (int j = 0; j < layerInput.Length; j++)
                    Console.Write("{0}  ", layerInput[j]);
                Console.WriteLine();
                Console.ReadKey();


                ------------------------- END DEBUGGING --------------------------------------------- */


                layers[l].FeedForward();


                /* ------------------------- DEBUGGING ---------------------------------------------

                // Display output layer-by-layer

                float[] layerOutput = new float[layers[l].Output.NumberOfUnits];
                CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                                layers[l].Output.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(layers[l].Output.NumberOfUnits * sizeof(float)),
                                                layerOutput,  // destination
                                                0,
                                                null,
                                                out CL.Event);
                CL.CheckErr(CL.Error, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerOutput");

                Console.WriteLine("\nLayer {0} ({1}) output activations:", l, layers[l].Type);
                for (int j = 0; j < layerOutput.Length; j++)
                        Console.Write("{0}  ", layerOutput[j]);
                Console.WriteLine();
                Console.ReadKey();


                ------------------------- END DEBUGGING --------------------------------------------- */

            }
        }
#else
        public void ForwardPass(float[] inputData)
        {

            //TODO: generalise to miniBatchSize > 1

            layers[0].Input.SetHost(inputData);
   
            // Run network forward
            for (int l = 0; l < nLayers; l++)
            {
                layers[l].FeedForward();
            }
        }
#endif




    }

    
}
