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
        private SoftMax outputLayer;

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

        /* // shouldn't need this
        public SoftMax OutputLayer
        {
            get { return outputLayer; }
            set { this.outputLayer = value; }
        }
        */
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
                { // TODO: add more error handling
                    case "Input":
                        {
                            if (!layers.Any()) // if list of layers is empty
                            {
                                newLayer.ID = 0;
                            }
                            else // list is not empty
                            {
                                throw new ArgumentException("You cannot add an InputLayer to a non-empty network.");
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
                    case "SoftMax":
                        {
                            if (layers.Last().Type != "FullyConnected")
                            {
                                throw new ArgumentException("You should only connect a SoftMax layer to a classifier (FullyConnected layer).");
                            }
                            else
                            {
                                newLayer.ID = layers.Last().ID + 1;
                                this.outputLayer = (SoftMax)newLayer;
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
                /*
                if (nLayers > 1)
                {
                    layers[newLayer.ID].ConnectTo(layers[newLayer.ID-1]); // connect last layer to second last
                    layers[newLayer.ID].InitializeParameters();
                }
                */
                Console.Write(" OK\n");
            }
            catch (Exception exception)
            {
                Console.WriteLine("\n\nException caught: \n{0}\n", exception);
            }
        }


        public void Setup(int miniBatchSize)
        {
            // Input layer (only setup output and buffers)
            layers[0].SetupOutput();
            layers[0].OutputNeurons.SetupBuffers(miniBatchSize);

            // Hidden layers and output layer
            for (int l = 1; l < nLayers; l++)
            {
                // 1. Setup input using output of previous layer.
                layers[l].ConnectTo(layers[l - 1]);

                // 2. Setup output neurons architecture using input and layer-specific properties
                // (e.g. filterSize and strideLenght in case of a ConvLayer)
                layers[l].SetupOutput();

                // 3. Allocate memory (if CPU) / buffers (if OpenCL) according to mini-batch size
                layers[l].OutputNeurons.SetupBuffers(miniBatchSize);

                // 4. Initialize layer's parameters
                layers[l].InitializeParameters();

                // 5. (extra) If using OpenCL, set global / local work group sizes for kernels
                layers[l].SetWorkGroups();
            }

            // Output layer only
            outputLayer.SetupOutputScores(miniBatchSize);
        }
        #endregion


        #region Training methods

        public void FeedData(DataSet dataSet, int[] iExamples)
        {
            int dataPointSize = dataSet.DataDimension;

            for (int m = 0; m < layers[0].OutputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                int iDataPoint = iExamples[m];
                
                OpenCLSpace.ClError = Cl.EnqueueCopyBuffer( OpenCLSpace.Queue,
                                                            dataSet.DataGPU[iDataPoint], // source
                                                            layers[0].OutputNeurons.ActivationsGPU, // destination
                                                            (IntPtr)0, // source offset (in bytes)
                                                            (IntPtr)(sizeof(float) * m * dataPointSize), // destination offset (in bytes)
                                                            (IntPtr)(sizeof(float) * dataPointSize),  // size of buffer to copy
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.FeedData Cl.clEnqueueReadBuffer inputData");
#else
                layers[0].OutputNeurons.SetHost(m, dataSet.GetDataPoint(iExamples[m]));
#endif
            }

#if DEBUGGING_STEPBYSTEP
            /* ------------------------- DEBUGGING --------------------------------------------- */


            // Display input layer-by-layer
            for (int m = 0; m < miniBatchSize; m++)
            {

                
#if OPENCL_ENABLED
                float[] inputData = new float[layers[0].OutputNeurons.NumberOfUnits];
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                            layers[0].OutputNeurons.ActivationsGPU[m], // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(layers[0].OutputNeurons.NumberOfUnits * sizeof(float)),
                                                            inputData,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.FeedData Cl.clEnqueueReadBuffer inputData");
#else
                double[] inputData = new double[layers[0].OutputNeurons.NumberOfUnits];
                inputData = layers[0].OutputNeurons.GetHost()[m];
#endif
                Console.WriteLine("\nLayer 0 (Input) output activations (mini-batch item {0}):", m);
                for (int j = 0; j < inputData.Length; j++)
                    Console.Write("{0}  ", inputData[j]);
                Console.WriteLine();
                Console.ReadKey();
            }

            /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif
        }

        /*
        public void FeedDatum(DataSet dataSet, int iExample)
        {
#if OPENCL_ENABLED
            layers[0].OutputNeurons.ActivationsGPU[0] = dataSet.DataGPU(iExample); // Copied by reference
#else
            layers[0].OutputNeurons.SetHost(0, dataSet.GetDataPoint(iExample));
#endif
        }
        */

        public void ForwardPass()
        {
            //TODO: generalise to miniBatchSize > 1

            // Run network forward
            for (int l = 1; l < nLayers; l++)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */


                // Display input layer-by-layer
                for (int m = 0; m < miniBatchSize; m++)
                {

                    
#if OPENCL_ENABLED
                    float[] layerInput = new float[layers[l].InputNeurons.NumberOfUnits];
                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                                layers[l].InputNeurons.ActivationsGPU[m], // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(layers[l].InputNeurons.NumberOfUnits * sizeof(float)),
                                                                layerInput,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");
#else
                    double[] layerInput = new double[layers[l].InputNeurons.NumberOfUnits];
                    layerInput = layers[l].InputNeurons.GetHost()[m];
#endif
                    Console.WriteLine("\nLayer {0} ({1}) input activations (mini-batch item {2}):", l, layers[l].Type, m);
                    for (int j = 0; j < layerInput.Length; j++)
                        Console.Write("{0}  ", layerInput[j]);
                    Console.WriteLine();
                    Console.ReadKey();
                }

                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif
                layers[l].FeedForward();

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display output layer-by-layer
                for (int m = 0; m < miniBatchSize; m++)
                {

                    
#if OPENCL_ENABLED
                    float[] layerOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                                layers[l].OutputNeurons.ActivationsGPU[m], // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(layers[l].OutputNeurons.NumberOfUnits * sizeof(float)),
                                                                layerOutput,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerOutput");
#else
                    double[] layerOutput = new double[layers[l].OutputNeurons.NumberOfUnits];
                    layerOutput = layers[l].OutputNeurons.GetHost()[m];
#endif
                    Console.WriteLine("\nLayer {0} ({1}) output activations (mini-batch item {2}):", l, layers[l].Type, m);
                    for (int j = 0; j < layerOutput.Length; j++)
                        Console.Write("{0}  ", layerOutput[j]);
                    Console.WriteLine();
                    Console.ReadKey();
                }

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
#if GRADIENT_CHECK
            learningRate = 0.0;
#endif

            for (int l = nLayers - 2; l > 0; l--) // propagate error signal backwards (layers L-2 to 1, i.e. second last to second)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display output delta layer-by-layer
                for (int m = 0; m < miniBatchSize; m++)
                {
                    
#if OPENCL_ENABLED
                    float[] deltaOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                                layers[l].OutputNeurons.DeltaGPU[m], // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(layers[l].OutputNeurons.NumberOfUnits * sizeof(float)),
                                                                deltaOutput,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.BackwardPass Cl.clEnqueueReadBuffer deltaOutput");
#else
                    double[] deltaOutput = new double[layers[l].OutputNeurons.NumberOfUnits];
                    deltaOutput = layers[l].OutputNeurons.DeltaHost[m];
#endif
                    Console.WriteLine("\nLayer {0} ({1}) output delta (mini-batch item {2}):", l, layers[l].Type, m);
                    for (int j = 0; j < deltaOutput.Length; j++)
                        Console.Write("{0}  ", deltaOutput[j]);
                    Console.WriteLine();
                    Console.ReadKey();
                }

                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif
                // 1. Update layer's parameters' change speed using gradient 
                layers[l].UpdateSpeeds(learningRate, momentumMultiplier);

                // 2. Backpropagate errors to previous layer (no need to do it for layer 1)
                if (l > 1)
                    layers[l].BackPropagate();
                

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display input delta layer-by-layer
                for (int m = 0; m < miniBatchSize; m++)
                {
                    
#if OPENCL_ENABLED
                    float[] deltaInput = new float[layers[l].InputNeurons.NumberOfUnits];
                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                                layers[l].InputNeurons.DeltaGPU[m], // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(layers[l].InputNeurons.NumberOfUnits * sizeof(float)),
                                                                deltaInput,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.BackwardPass Cl.clEnqueueReadBuffer deltaInput");
#else
                    double[] deltaInput = new double[layers[l].InputNeurons.NumberOfUnits];
                    deltaInput = layers[l].InputNeurons.DeltaHost[m];
#endif
                    Console.WriteLine("\nLayer {0} ({1}) input delta (mini-batch item {2}):", l, layers[l].Type, m);
                    for (int j = 0; j < deltaInput.Length; j++)
                        Console.Write("{0}  ", deltaInput[j]);
                    Console.WriteLine();
                    Console.ReadKey();
                }

                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

#if GRADIENT_CHECK
                // do nothing
#else
                // 3. Update layer's parameters
                layers[l].UpdateParameters();
#endif
            }
        }


        public void CrossEntropyGradient(DataSet DataSet, int[] iMiniBatch)
        {

            for (int m = 0; m < layers.Last().OutputNeurons.MiniBatchSize; m++)
            {
                int iDataPoint = iMiniBatch[m];
                int trueLabel = DataSet.Labels[iDataPoint];

                double[] crossEntropyGradient = layers.Last().OutputClassScores[m];
                crossEntropyGradient[trueLabel] -= 1.0F;

                // now write gradient to input neurons of softmax layer (i.e. to output neurons of classifier)
#if OPENCL_ENABLED
                float[] floatCrossEntropyGradient = new float[crossEntropyGradient.Length];
                for (int c = 0; c < crossEntropyGradient.Length; c++)
                {
                    floatCrossEntropyGradient[c] = (float)crossEntropyGradient[c];
                }

                OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue, 
                                                            layers.Last().InputNeurons.DeltaGPU, 
                                                            OpenCL.Net.Bool.True,
                                                            (IntPtr) (sizeof(float) * m * crossEntropyGradient.Length), 
                                                            (IntPtr) (sizeof(float) * crossEntropyGradient.Length),
                                                            floatCrossEntropyGradient, 
                                                            0, 
                                                            null, 
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NetworkTrainer.CrossEntropyGradient(): Cl.EnqueueWriteBuffer");


                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                layers.Last().InputNeurons.DeltaHost[m] = crossEntropyGradient;
#endif

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif

            }

        }
        #endregion

    } 
}
