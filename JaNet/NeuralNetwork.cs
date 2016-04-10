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
        #region Fields

        private List<Layer> layers;
        private int nLayers;

        #endregion


        #region Properties

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
            Console.WriteLine("New empty network created.");
            this.layers = new List<Layer>(); // empty list of layers
            this.nLayers = 0;
        }

        /// <summary>
        /// Add layer to NeuralNetwork instance.
        /// </summary>
        /// <param name="newLayer"></param>
        public void AddLayer(Layer newLayer)
        {
            // Some error-handling
            try
            {

                if (!layers.Any() && newLayer.Type != "Input")
                {
                    throw new ArgumentException("Need to add an InputLayer as 0th layer of the network.");
                }

                switch (newLayer.Type)
                {
                    case "Input":
                        {
                            if (!layers.Any()) // if list of layers is empty
                            {
                                newLayer.ID = 0;
                            }
                            else // list is not empty
                            {
                                throw new ArgumentException("Adding an InputLayer to a non-empty network.");
                            }
                            break;
                        }
                    case "Convolutional":
                        {
                            if (layers.Last().Type != "Input" & layers.Last().Type != "Convolutional" & layers.Last().Type != "Pooling")
                            {
                                throw new ArgumentException("You should only connect a ConvolutionalLayer to an InputLayer, a ConvolutionalLayer or a PoolingLayer");
                            }
                            else
                            {
                                newLayer.ID = layers.Last().ID + 1;
                            }
                            break;
                        }
                    case "Pooling":
                        {
                            if (layers.Last().Type != "Convolutional")
                            {
                                throw new ArgumentException("You should only connect a PoolingLayer to a ConvolutionalLayer");
                            }
                            else
                            {
                                newLayer.ID = layers.Last().ID + 1;
                            }
                            break;
                        }
                    default: // valid connection
                        newLayer.ID = layers.Last().ID + 1;
                        break;
                }

                Console.Write("\tAdding layer [" + newLayer.ID + "]: " + newLayer.Type + "...");
                layers.Add(newLayer);
                nLayers++;

                if (nLayers == 1)
                {
                    Console.Write(" OK\n");
                }
                else
                {
                    layers[newLayer.ID].ConnectTo(layers[newLayer.ID-1]); // connect last layer to second last
                    layers[newLayer.ID].InitializeParameters();
                    Console.Write(" OK\n");
                }

            }
            catch (Exception exception)
            {
                Console.WriteLine("\n\nException caught: \n{0}\n", exception);
            }
        }


        #endregion


        #region Training methods

        public void FeedData(DataSet dataSet, int iDataPoint)
        {
#if OPENCL_ENABLED
            layers[0].OutputNeurons.ActivationsGPU = dataSet.DataGPU(iDataPoint); // Copied by reference
#else
            layers[0].OutputNeurons.SetHost(dataSet.GetDataPoint(iDataPoint));
#endif
        }


        public void ForwardPass()
        {
            //TODO: generalise to miniBatchSize > 1

            // Run network forward
            for (int l = 1; l < nLayers; l++)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display input layer-by-layer

                float[] layerInput = new float[layers[l].InputNeurons.NumberOfUnits];
#if OPENCL_ENABLED
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                layers[l].InputNeurons.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(layers[l].InputNeurons.NumberOfUnits * sizeof(float)),
                                                layerInput,  // destination
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");
#else
                layerInput = layers[l].InputNeurons.GetHost();
#endif
                Console.WriteLine("\nLayer {0} ({1}) input activations:",l , layers[l].Type);
                for (int j = 0; j < layerInput.Length; j++)
                    Console.Write("{0}  ", layerInput[j]);
                Console.WriteLine();
                Console.ReadKey();


                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif
                layers[l].FeedForward();

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display output layer-by-layer

                float[] layerOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
#if OPENCL_ENABLED
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                layers[l].OutputNeurons.ActivationsGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(layers[l].OutputNeurons.NumberOfUnits * sizeof(float)),
                                                layerOutput,  // destination
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerOutput");
#else
                layerOutput = layers[l].OutputNeurons.GetHost();
#endif
                Console.WriteLine("\nLayer {0} ({1}) output activations:", l, layers[l].Type);
                for (int j = 0; j < layerOutput.Length; j++)
                        Console.Write("{0}  ", layerOutput[j]);
                Console.WriteLine();
                Console.ReadKey();


                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

            }
        }

        /// <summary>
        /// Run network backwards, propagating the gradient backwards and also updating parameters. 
        /// Requires that gradient has ALREADY BEEN WRITTEN in network.Layers[nLayers-1].InputNeurons.Delta
        /// </summary>
        public void BackwardPass(double learningRate, double momentumMultiplier)
        {
            for (int l = nLayers - 2; l > 0; l--) // propagate error signal backwards (layers L-2 to 1)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display output layer-by-layer
                float[] deltaOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
#if OPENCL_ENABLED
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                layers[l].OutputNeurons.DeltaGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(layers[l].OutputNeurons.NumberOfUnits * sizeof(float)),
                                                deltaOutput,  // destination
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.BackwardPass Cl.clEnqueueReadBuffer deltaOutput");
#else
                deltaOutput = layers[l].OutputNeurons.DeltaHost;
#endif
                Console.WriteLine("\nLayer {0} ({1}) output delta:", l, layers[l].Type);
                for (int j = 0; j < deltaOutput.Length; j++)
                    Console.Write("{0}  ", deltaOutput[j]);
                Console.WriteLine();
                Console.ReadKey();


                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif
                if (l > 1) // no need to backprop layer 1
                {
                    layers[l].BackPropagate();
                }

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display output layer-by-layer
                float[] deltaInput = new float[layers[l].InputNeurons.NumberOfUnits];
#if OPENCL_ENABLED
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                layers[l].InputNeurons.DeltaGPU, // source
                                                Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(layers[l].InputNeurons.NumberOfUnits * sizeof(float)),
                                                deltaInput,  // destination
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.BackwardPass Cl.clEnqueueReadBuffer deltaInput");
#else
                deltaInput = layers[l].InputNeurons.DeltaHost;
#endif
                Console.WriteLine("\nLayer {0} ({1}) input delta:", l, layers[l].Type);
                for (int j = 0; j < deltaInput.Length; j++)
                    Console.Write("{0}  ", deltaInput[j]);
                Console.WriteLine();
                Console.ReadKey();


                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif
                layers[l].UpdateParameters(learningRate, momentumMultiplier);
            }
        }

        #endregion

    } 
}
