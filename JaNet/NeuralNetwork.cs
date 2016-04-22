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

        private InputLayer inputLayer;
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

        public InputLayer InputLayer
        {
            get { return inputLayer; }
            set { this.inputLayer = value; }
        }

        public SoftMax OutputLayer
        {
            get { return outputLayer; }
            set { this.outputLayer = value; }
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
                { // TODO: add more error handling
                    case "Input":
                        {
                            if (!layers.Any()) // if list of layers is empty
                            {
                                newLayer.ID = 0;
                                this.inputLayer = (InputLayer)newLayer;
                            }
                            else // list is not empty
                            {
                                throw new ArgumentException("You cannot add an InputLayer to a non-empty network.");
                            }
                            break;
                        }
                    case "Pooling":
                        {
                            if (layers.Last().Type != "ReLU")
                            {
                                throw new ArgumentException("You should only connect a PoolingLayer to a non-linearity layer.");
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
                        {
                            newLayer.ID = layers.Last().ID + 1;
                            break;
                        }
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


        #region Methods

        public void ForwardPass()
        {
            //TODO: generalise to miniBatchSize > 1

            // Run network forward
            for (int l = 1; l < nLayers; l++)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */
                int miniBatchSize = layers[0].OutputNeurons.MiniBatchSize;
                if (l < nLayers - 1)
                {
#if OPENCL_ENABLED
                    float[] layerInputAll = new float[layers[l].InputNeurons.NumberOfUnits * miniBatchSize];
                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                                layers[l].InputNeurons.ActivationsGPU, // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(layers[l].InputNeurons.NumberOfUnits * miniBatchSize * sizeof(float)),
                                                                layerInputAll,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");
#endif
                    // Display input layer-by-layer
                    Console.WriteLine("\nLayer {0} ({1}) input activations:", l, layers[l].Type);
                    for (int m = 0; m < miniBatchSize; m++)
                    {


#if OPENCL_ENABLED
                        float[] layerInput = new float[layers[l].InputNeurons.NumberOfUnits];
                        Array.Copy(layerInputAll, m * layers[l].InputNeurons.NumberOfUnits, layerInput, 0, layers[l].InputNeurons.NumberOfUnits);
#else
                    double[] layerInput = new double[layers[l].InputNeurons.NumberOfUnits];
                    layerInput = layers[l].InputNeurons.GetHost()[m];
#endif
                        Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                        for (int j = 0; j < layerInput.Length; j++)
                            Console.Write("{0}  ", layerInput[j]);
                        Console.WriteLine();
                        Console.ReadKey();
                    }
                }
                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif
                layers[l].FeedForward();

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display output layer-by-layer


                /* ------------------------- DEBUGGING --------------------------------------------- */
                if (l < nLayers - 1)
                {
#if OPENCL_ENABLED
                    float[] layerOutputAll = new float[layers[l].OutputNeurons.NumberOfUnits * miniBatchSize];
                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                                layers[l].OutputNeurons.ActivationsGPU, // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(layers[l].OutputNeurons.NumberOfUnits * miniBatchSize * sizeof(float)),
                                                                layerOutputAll,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");
#endif
                    Console.WriteLine("\nLayer {0} ({1}) output activations:", l, layers[l].Type);
                    for (int m = 0; m < miniBatchSize; m++)
                    {

                        
#if OPENCL_ENABLED
                        float[] layerOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
                        Array.Copy(layerOutputAll, m * layers[l].OutputNeurons.NumberOfUnits, layerOutput, 0, layers[l].OutputNeurons.NumberOfUnits);
#else
                    double[] layerOutput = new double[layers[l].OutputNeurons.NumberOfUnits];
                    layerOutput = layers[l].OutputNeurons.GetHost()[m];
#endif
                        Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                        for (int j = 0; j < layerOutput.Length; j++)
                            Console.Write("{0}  ", layerOutput[j]);
                        Console.WriteLine();
                        Console.ReadKey();
                    }
                }
                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

            }
        }

        /// <summary>
        /// Run network backwards, propagating the gradient backwards and also updating parameters. 
        /// Requires that gradient has ALREADY BEEN WRITTEN in network.Layers[nLayers-1].InputNeurons.Delta
        /// </summary>
        public void BackwardPass(double learningRate, double momentumMultiplier, double weightDecayCoeff)
        {
#if GRADIENT_CHECK
            learningRate = 0.0;
#endif

            for (int l = nLayers - 2; l > 0; l--) // propagate error signal backwards (layers L-2 to 1, i.e. second last to second)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING ---------------------------------------------

                // Display output delta  layer-by-layer
                int miniBatchSize = layers[0].OutputNeurons.MiniBatchSize;

#if OPENCL_ENABLED
               
                float[] deltaOutputAll = new float[layers[l].OutputNeurons.NumberOfUnits * miniBatchSize];
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                            layers[l].OutputNeurons.DeltaGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(layers[l].OutputNeurons.NumberOfUnits * miniBatchSize * sizeof(float)),
                                                            deltaOutputAll,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer deltaOutputAll");
#endif
                for (int m = 0; m < miniBatchSize; m++)
                {

                    Console.WriteLine("\n --- Mini-batch item {0} -----", m);
#if OPENCL_ENABLED
                    float[] deltaOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
                    Array.Copy(deltaOutputAll, m * layers[l].OutputNeurons.NumberOfUnits, deltaOutput, 0, layers[l].OutputNeurons.NumberOfUnits);
#else
                    double[] deltaOutput = new double[layers[l].OutputNeurons.NumberOfUnits];
                    deltaOutput = layers[l].OutputNeurons.DeltaHost[m];
#endif
                    Console.WriteLine("\nLayer {0} ({1}) output delta:", l, layers[l].Type);
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

                // Display input delta  layer-by-layer

                int miniBatchSize = layers[0].OutputNeurons.MiniBatchSize;
#if OPENCL_ENABLED
                float[] deltaInputAll = new float[layers[l].InputNeurons.NumberOfUnits * miniBatchSize];
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                            layers[l].InputNeurons.DeltaGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(layers[l].InputNeurons.NumberOfUnits * miniBatchSize * sizeof(float)),
                                                            deltaInputAll,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer deltaInputAll");
#endif
                Console.WriteLine("\nLayer {0} ({1}) backpropagated delta:", l, layers[l].Type);
                for (int m = 0; m < miniBatchSize; m++)
                {

                    
#if OPENCL_ENABLED
                    float[] deltaInput = new float[layers[l].InputNeurons.NumberOfUnits];
                    Array.Copy(deltaInputAll, m * layers[l].InputNeurons.NumberOfUnits, deltaInput, 0, layers[l].InputNeurons.NumberOfUnits);
#else
                    double[] deltaInput = new double[layers[l].InputNeurons.NumberOfUnits];
                    deltaInput = layers[l].InputNeurons.DeltaHost[m];
#endif
                    Console.WriteLine("\n --- Mini-batch item {0} -----", m);
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
                layers[l].UpdateParameters(weightDecayCoeff);
#endif
            }
        }


        public void CrossEntropyGradient(DataSet DataSet, int[] iMiniBatch)
        {
            double[] crossEntropyGradientBatch = new double[layers.Last().NInputUnits * iMiniBatch.Length];

#if DEBUGGING_STEPBYSTEP
            Console.WriteLine("\n CLASS SCORES");
#endif
            for (int m = 0; m < iMiniBatch.Length; m++)
            {
                int iDataPoint = iMiniBatch[m];
                int trueLabel = DataSet.Labels[iDataPoint];

                double[] crossEntropyGradient = outputLayer.OutputClassScores[m];

#if DEBUGGING_STEPBYSTEP
                Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                for (int j = 0; j < outputLayer.OutputClassScores[m].Length; j++)
                    Console.Write("{0}  ", outputLayer.OutputClassScores[m][j]);
                Console.WriteLine();
                Console.ReadKey();
#endif
                crossEntropyGradient[trueLabel] -= 1.0;

                crossEntropyGradient.CopyTo(crossEntropyGradientBatch, m * crossEntropyGradient.Length);
            }

            // now write gradient to input neurons of softmax layer (i.e. to output neurons of classifier)
#if OPENCL_ENABLED
            float[] floatCrossEntropyGradient = new float[crossEntropyGradientBatch.Length];
            for (int c = 0; c < crossEntropyGradientBatch.Length; c++)
            {
                floatCrossEntropyGradient[c] = (float)crossEntropyGradientBatch[c];
            }

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue, 
                                                        layers.Last().InputNeurons.DeltaGPU, 
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr) 0, 
                                                        (IntPtr) (sizeof(float) * floatCrossEntropyGradient.Length),
                                                        floatCrossEntropyGradient, 
                                                        0, 
                                                        null, 
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NetworkTrainer.CrossEntropyGradient(): Cl.EnqueueWriteBuffer");


            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            for (int m = 0; m < iMiniBatch.Length; m++)
            {
                Array.Copy(crossEntropyGradientBatch, m * layers.Last().NInputUnits, layers.Last().InputNeurons.DeltaHost[m], 0, layers.Last().NInputUnits);
            }
#endif

        }
        #endregion

    } 
}
