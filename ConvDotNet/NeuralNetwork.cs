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
    [Serializable]
    class NeuralNetwork
    {
        #region Fields

        private string name;

        private List<Layer> layers;
        private int nLayers;

        private InputLayer inputLayer;
        private SoftMax outputLayer;

        private double dropoutFC;
        private double dropoutConv;
        private double dropoutInput;

        private int inputChannels;

        #endregion


        #region Properties

        public string Name
        {
            get { return name; }
            set { name = value; }
        }

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

        public double DropoutFC
        {
            get { return dropoutFC; }
            set { throw new InvalidOperationException("Use method Set(''DropoutFC'', <value>) to set field 'dropoutFC'"); }
        }

        public double DropoutConv
        {
            get { return dropoutConv; }
            set { throw new InvalidOperationException("Use method Set(''DropoutConv'', <value>) to set field 'dropoutConv'"); }
        }

        public double DropoutInput
        {
            get { return dropoutInput; }
            set { throw new InvalidOperationException("Use method Set(''DropoutInput'', <value>) to set field 'dropoutInput'"); }
        }

        public int InputChannels
        {
            get { return inputChannels; }
            set { this.inputChannels = value; }
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
        /// NeuralNetwork class constructor.
        /// </summary>
        /// <param name="NetworkName"></param>
        public NeuralNetwork(string NetworkName)
        {
            Console.WriteLine("New empty network created.");
            this.layers = new List<Layer>(); // empty list of layers
            this.nLayers = 0;
            this.Name = NetworkName;
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
                                this.inputChannels = newLayer.InputDepth;
                            }
                            else // list is not empty
                            {
                                throw new ArgumentException("You cannot add an InputLayer to a non-empty network.");
                            }
                            break;
                        }
                    case "Pooling":
                        {
                            if (layers.Last().Type != "ReLU" && layers.Last().Type != "ELU")
                            {
                                throw new ArgumentException("Perhaps you forgot to add a non-linearity?");
                            }
                            else
                            {
                                newLayer.ID = layers.Last().ID + 1;
                            }
                            break;
                        }
                    case "SoftMax":
                        {
                            newLayer.ID = layers.Last().ID + 1;
                            this.outputLayer = (SoftMax)newLayer;
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
                Console.Write(" OK\n");
            }
            catch (Exception exception)
            {
                Console.WriteLine("\n\nException caught: \n{0}\n", exception);
            }
        }


        public void InitializeParameters(string Option)
        {
            // check argument
            if (Option != "random" && Option != "load")
                throw new ArgumentException("Pass either ''random'' (to initialize new parameters randomly via sampling), or ''load'' (to load parameters from saved network)");
            for (int l = 1; l < nLayers; l++)
            {
                layers[l].InitializeParameters(Option);
            }
        }




        public void Set(string ArgumentString, object value)
        {
            switch (ArgumentString)
            {
                case "MiniBatchSize":
                    {
                        int miniBatchSize = (int)value;

                        // Input layer (only setup output and buffers)
                        layers[0].SetupOutput();
                        layers[0].OutputNeurons.SetupBuffers(miniBatchSize);
                        layers[0].SetWorkGroups();

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

                            // 4. (extra) If using OpenCL, set global / local work group sizes for kernels
                            layers[l].SetWorkGroups();
                        }

                        // Output layer only
                        outputLayer.SetupOutputScores(miniBatchSize);

                        break;
                    }
                case "DropoutFC":
                    {
                        dropoutFC = (double)value;
                        for (int l = 1; l < nLayers - 2; l++) // excluding input layer, final FC layer and softmax
                        {
                            if (layers[l].Type == "FullyConnected")
                                layers[l].DropoutParameter = dropoutFC;
                        }

                        // No dropout in last FC layer (just make sure)
                        layers[nLayers - 2].DropoutParameter = 1;

                        break;
                    }
                case "DropoutConv":
                    {
                        dropoutConv = (double)value;
                        for (int l = 1; l < nLayers - 2; l++) // excluding input layer, final FC layer and softmax
                        {
                            if (layers[l].Type == "Convolutional")
                                layers[l].DropoutParameter = dropoutConv;
                        }

                        break;
                    }
                case "DropoutInput":
                    {
                        dropoutInput = (double)value;
                        inputLayer.DropoutParameter = dropoutInput;
                        break;
                    }
                case "EpochBeginning":
                    {
                        if ((bool)value == true)
                        {
                            for (int l = 1; l < nLayers - 1; ++l)
                            {
                                if (layers[l].Type == "BatchNormFC" || layers[l].Type == "BatchNormConv")
                                    layers[l].IsEpochBeginning = true;
                            }
                        }
                        else
                            throw new ArgumentException("Wrong argument passed to NeuralNetwork.Set()");

                        break;
                    }
                case "Training":
                    {
                        if ((bool)value == true)
                        {
                            for (int l = 1; l < nLayers - 1; ++l)
                            {
                                if (layers[l].Type == "BatchNormFC" || layers[l].Type == "BatchNormConv")
                                {
                                    layers[l].IsTraining = true;
                                    layers[l].IsPreInference = false;
                                    layers[l].IsInference = false;
                                }
                            }
                        }
                        else
                            throw new ArgumentException("Only <true> can be passed to Set(''Training'', <arg>)");
                        break;
                    }
                case "PreInference":
                    {
                        if ((bool)value == true)
                        {
                            for (int l = 1; l < nLayers - 1; ++l)
                            {
                                if (layers[l].Type == "BatchNormFC" || layers[l].Type == "BatchNormConv")
                                {
                                    layers[l].IsTraining = false;
                                    layers[l].IsPreInference = true;
                                    layers[l].IsInference = false;
                                }
                            }
                        }
                        else
                            throw new ArgumentException("Only <true> can be passed to Set(''PreInference'', <arg>)");
                        break;
                    }
                case "Inference":
                    {
                        if ((bool)value == true)
                        {
                            for (int l = 1; l < nLayers - 1; ++l)
                            {
                                if (layers[l].Type == "BatchNormFC" || layers[l].Type == "BatchNormConv")
                                {
                                    layers[l].IsTraining = false;
                                    layers[l].IsPreInference = false;
                                    layers[l].IsInference = true;
                                }
                            }
                        }
                        else
                            throw new ArgumentException("Only <true> can be passed to Set(''Inference'', <arg>)");
                        break;
                    }
                default:
                    throw new ArgumentException("Wrong argument passed to NeuralNetwork.Set()");

            }
        }

        #endregion


        #region Methods

        public void ForwardPass(object StartPoint, object EndPoint)
        {
            int iStartLayer, iEndLayer;

            if (StartPoint.GetType() == typeof(string))
            {
                if (StartPoint.ToString() == "beginning")
                    iStartLayer = 1;
                else
                    throw new ArgumentException("First argument: pass either ''beginning'', or an integer corresponding to starting layer.");
            }
            else if (StartPoint.GetType() == typeof(int))
                iStartLayer = (int)StartPoint;
            else
                throw new ArgumentException("First argument <StartPoint> is invalid.");

            if (EndPoint.GetType() == typeof(string))
            {
                if (EndPoint.ToString() == "end")
                    iEndLayer = nLayers;
                else
                    throw new ArgumentException("Second argument: pass either ''end'', or an integer corresponding to end layer.");
            }
            else if (EndPoint.GetType() == typeof(int))
                iEndLayer = (int)EndPoint;
            else
                throw new ArgumentException("Second argument <EndPoint> is invalid.");


            // Run network forward
            for (int l = iStartLayer; l < iEndLayer; l++)
            {

#if DEBUGGING_STEPBYSTEP
                /* ------------------------- DEBUGGING ---------------------------------------------*/
                int miniBatchSize = layers[0].OutputNeurons.MiniBatchSize;
                if (l < nLayers - 1)
                {

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

                    // Display input layer-by-layer
                    Console.WriteLine("\nLayer {0} ({1}) input activations:", l, layers[l].Type);
                    for (int m = 0; m < miniBatchSize; m++)
                    {



                        float[] layerInput = new float[layers[l].InputNeurons.NumberOfUnits];
                        Array.Copy(layerInputAll, m * layers[l].InputNeurons.NumberOfUnits, layerInput, 0, layers[l].InputNeurons.NumberOfUnits);

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
                //int miniBatchSize = layers[0].OutputNeurons.MiniBatchSize;

                if (l < nLayers-1)
                {

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

                    Console.WriteLine("\nLayer {0} ({1}) output activations:", l, layers[l].Type);
                    for (int m = 0; m < miniBatchSize; m++)
                    {

                        

                        float[] layerOutput = new float[layers[l].OutputNeurons.NumberOfUnits];
                        Array.Copy(layerOutputAll, m * layers[l].OutputNeurons.NumberOfUnits, layerOutput, 0, layers[l].OutputNeurons.NumberOfUnits);

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



            /*
            using (System.IO.StreamWriter classScoresFile = new System.IO.StreamWriter(@"C:\Users\jacopo\Desktop\ClassScores_08.txt", true))
            {

                for (int m = 0; m < layers[0].OutputNeurons.MiniBatchSize; m++)
                {
                    double[] outputScores = outputLayer.OutputClassScores[m];

                    for (int j = 0; j < outputScores.Length; j++)
                        classScoresFile.Write(outputScores[j].ToString() + "\t");
                    classScoresFile.WriteLine();
                }
            }
            */


#if DEBUGGING_STEPBYSTEP
            Console.WriteLine("Class scores (softmax activation):");
            for (int m = 0; m < layers[0].OutputNeurons.MiniBatchSize; m++)
            {
                double[] outputScores = outputLayer.OutputClassScores[m];

                Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                for (int j = 0; j < outputScores.Length; j++)
                    Console.Write("{0}  ", (float)outputScores[j]);
                Console.WriteLine();
                Console.ReadKey();
            }
#endif
        }

        /// <summary>
        /// Run network backwards, propagating the gradient backwards and also updating parameters. 
        /// Requires that gradient has ALREADY BEEN WRITTEN in network.Layers[nLayers-1].InputNeurons.Delta
        /// </summary>
        public void BackwardPass(double learningRate, double momentumMultiplier, double weightDecayCoeff, double weightMaxNorm)
        {

            for (int l = nLayers - 2; l > 0; l--) // propagate error signal backwards (layers L-2 to 1, i.e. second last to second)
            {
                // 1. Update layer's parameters' change speed using gradient 
                layers[l].UpdateSpeeds(learningRate, momentumMultiplier, weightDecayCoeff);

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

                    

                    float[] deltaInput = new float[layers[l].InputNeurons.NumberOfUnits];
                    Array.Copy(deltaInputAll, m * layers[l].InputNeurons.NumberOfUnits, deltaInput, 0, layers[l].InputNeurons.NumberOfUnits);

                    Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                    for (int j = 0; j < deltaInput.Length; j++)
                        Console.Write("{0}  ", deltaInput[j]);
                    Console.WriteLine();
                    Console.ReadKey();
                }

                /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

                // 3. Update layer's parameters
                layers[l].UpdateParameters(weightMaxNorm);
            }
        }





        public void CrossEntropyGradient(DataSet DataSet, int[] iMiniBatch)
        {
            float[] crossEntropyGradientBatch = new float[iMiniBatch.Length * DataSet.NumberOfClasses];
            int nClasses = DataSet.NumberOfClasses;

            for (int m = 0; m < iMiniBatch.Length; m++)
            {
                int iDataPoint = iMiniBatch[m];
                int trueLabel = DataSet.Labels[iDataPoint];

                double[] crossEntropyGradient = new double[nClasses];
                Array.Copy(outputLayer.OutputClassScores[m], crossEntropyGradient, nClasses);
                crossEntropyGradient[trueLabel] -= 1.0;

                for (int c = 0; c < nClasses; c++)
                {
                    crossEntropyGradientBatch[m * DataSet.NumberOfClasses + c] = (float)crossEntropyGradient[c];
                }

            }

            // now write gradient to input neurons of softmax layer (i.e. to output neurons of classifier)


            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue, 
                                                        layers.Last().InputNeurons.DeltaGPU, 
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr) 0, 
                                                        (IntPtr) (sizeof(float) * crossEntropyGradientBatch.Length),
                                                        crossEntropyGradientBatch, 
                                                        0, 
                                                        null, 
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NetworkTrainer.CrossEntropyGradient(): Cl.EnqueueWriteBuffer");


            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

        }

        #endregion


        #region SaveWeights

        public void SaveWeights(string whichLayer, string outputDirPath)
        {
            int n;
            if (whichLayer == "all")
                n = nLayers;
            else if (whichLayer == "first")
                n = 1;
            else
                throw new ArgumentException("First argument must be either ''first'' or ''all''");


            for (int iLayer = 1; iLayer <= n; ++iLayer)
            {
                if (layers[iLayer].Type == "Convolutional")
                {
                    string outputFilePath = outputDirPath + name + "_layer" + iLayer.ToString() + "_convolutional_filters.txt";

                    Mem filtersGPU = layers[iLayer].WeightsGPU;

                    int nFilters = layers[iLayer].OutputDepth;
                    int inputDepth = layers[iLayer].InputDepth;
                    int filterSize = layers[iLayer].FilterSize;

                    int nParameters = nFilters * inputDepth * filterSize * filterSize;

                    float[] filters = new float[nParameters];

                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                                filtersGPU, // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(sizeof(float) * nParameters),
                                                                filters,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer filtersGPU");

                    OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                    OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                    using (System.IO.StreamWriter outputFile = new System.IO.StreamWriter(outputFilePath))
                    {
                        foreach (float filterValue in filters)
                        {
                            outputFile.WriteLine(filterValue.ToString());
                        }
                        Console.WriteLine("Weights of layer " + iLayer.ToString() + " (convolutional) saved to file" + outputFilePath);
                    }
                }
                else if (layers[iLayer].Type == "FullyConnected")
                {
                    string outputFilePath = outputDirPath + name + "_layer" + iLayer.ToString() + "_fullyConnected_weights.txt";

                    Mem weightsGPU = layers[iLayer].WeightsGPU;

                    int nOutputUnits = layers[iLayer].NOutputUnits;
                    int nInputUnits = layers[iLayer].NInputUnits;

                    int nParameters = nOutputUnits * nInputUnits;

                    float[] weights = new float[nParameters];

                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                                weightsGPU, // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(sizeof(float) * nParameters),
                                                                weights,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer weightsGPU");

                    OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                    OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                    using (System.IO.StreamWriter outputFile = new System.IO.StreamWriter(outputFilePath))
                    {
                        foreach (float weightValue in weights)
                        {
                            outputFile.WriteLine(weightValue.ToString());
                        }
                        Console.WriteLine("Weights of layer " + iLayer.ToString() + " (fully connected) saved to file" + outputFilePath);
                    }
                }
                
            }
        }
        #endregion
    } 
}
