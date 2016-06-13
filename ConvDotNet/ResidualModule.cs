using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net;

namespace Conv.NET
{
    [Serializable]
    public class ResidualModule : Layer
    {
        #region Fields

        private ConvolutionalLayer convolutionalLayer1;
        private ReLU nonlinearityReLU; //TODO: generalize "Nonlinearity" layer
        private ELU nonlinearityELU;
        private ConvolutionalLayer convolutionalLayer2;

        private int filterSize;
        private int nFilters;
        private int strideLength;
        private int zeroPadding;
        private string nonlinearityType;

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr;


        #endregion


        #region Properties

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of fully connected layer type. Specify number of units as argument.
        /// </summary>
        /// <param name="nUnits"></param>
        public ResidualModule(int FilterSize, int NumberOfFilters, int StrideLength, int ZeroPadding, string NonlinearityType)
        {
            this.type = "ResidualModule";

            filterSize = FilterSize;
            nFilters = NumberOfFilters;
            strideLength = StrideLength;
            zeroPadding = ZeroPadding;

            if (NonlinearityType != "ReLU" && NonlinearityType != "ELU")
                throw new ArgumentException("Only ReLU or ELU nonlinearities are currently supported in ResidualModules");

            nonlinearityType = NonlinearityType;

            convolutionalLayer1 = new ConvolutionalLayer(filterSize, nFilters, strideLength, zeroPadding);
            if (NonlinearityType == "ReLU")
                nonlinearityReLU = new ReLU();
            else if (NonlinearityType == "ELU")
                nonlinearityELU = new ELU(1.0f);
            convolutionalLayer2 = new ConvolutionalLayer(filterSize, nFilters, strideLength, zeroPadding);
        }


        public override void SetupOutput()
        {
            // TODO: should include ConnectTo() , SetupOutput(), OutputNeurons.SetupBuffers(miniBatchSize), and SetWorkGroups() of member layers

            //______________________________________________________________________________________________________
            // Setup first convolutional layer

            convolutionalLayer1.InputNeurons = this.InputNeurons; // assignment is by reference! 
            convolutionalLayer1.NInputUnits = this.NInputUnits;
            convolutionalLayer1.InputWidth = this.InputWidth;
            convolutionalLayer1.InputHeight = this.InputHeight;
            convolutionalLayer1.InputDepth = this.InputDepth;

            convolutionalLayer1.SetupOutput();
            convolutionalLayer1.OutputNeurons.SetupBuffers(convolutionalLayer1.InputNeurons.MiniBatchSize);
            convolutionalLayer1.SetWorkGroups();

            //______________________________________________________________________________________________________
            // Setup nonlinearity
            if (nonlinearityType == "ReLU")
            {
                nonlinearityReLU.ConnectTo(convolutionalLayer1);
                nonlinearityReLU.SetupOutput();
                nonlinearityReLU.OutputNeurons.SetupBuffers(nonlinearityReLU.InputNeurons.MiniBatchSize);
                nonlinearityReLU.SetWorkGroups();

                convolutionalLayer2.ConnectTo(nonlinearityReLU);
            }
            else if (nonlinearityType == "ELU")
            {
                nonlinearityELU.ConnectTo(convolutionalLayer1);
                nonlinearityELU.SetupOutput();
                nonlinearityELU.OutputNeurons.SetupBuffers(nonlinearityELU.InputNeurons.MiniBatchSize);
                nonlinearityELU.SetWorkGroups();

                convolutionalLayer2.ConnectTo(nonlinearityELU);
            }
            //______________________________________________________________________________________________________
            // Setup second convolutional layer

            convolutionalLayer2.SetupOutput();
            //convolutionalLayer2.OutputNeurons.SetupBuffers(convolutionalLayer2.InputNeurons.MiniBatchSize);
            convolutionalLayer2.SetWorkGroups();

            //______________________________________________________________________________________________________
            // Setup output neurons
            this.outputWidth = convolutionalLayer2.OutputWidth;
            this.outputHeight = convolutionalLayer2.OutputHeight;
            this.outputDepth = convolutionalLayer2.OutputDepth;

            this.nOutputUnits = outputDepth * outputWidth * outputHeight;
            this.outputNeurons = convolutionalLayer2.OutputNeurons;
        }

        public override void SetWorkGroups()
        {
            /*
             * 
            // Local
            int optBaseRatio = OpenCLSpace.OPTIMAL_GROUP_SIZE / OpenCLSpace.BASE_GROUP_SIZE;
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)optBaseRatio, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };

            // Global
            int smallestMultiple0 = (int)(optBaseRatio * Math.Ceiling((double)(inputNeurons.MiniBatchSize) / (double)optBaseRatio));
            int smallestMultiple1 = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1 };
             */

            // Local
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };

            // Global
            int smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits * inputNeurons.MiniBatchSize) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };
        }

        public override void InitializeParameters(string Option)
        {
            base.InitializeParameters(Option); // makes sure this method is only call AFTER "SetupOutput()"

            convolutionalLayer1.InitializeParameters(Option);
            convolutionalLayer2.InitializeParameters(Option);
        }

        public override void CopyBuffersToHost()
        {
            convolutionalLayer1.CopyBuffersToHost();
            convolutionalLayer2.CopyBuffersToHost();
        }


        #endregion


        #region Methods

        public override void FeedForward()
        {
            convolutionalLayer1.FeedForward();


            /*

            float[] conv1outputAll = new float[convolutionalLayer1.OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        convolutionalLayer1.OutputNeurons.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(convolutionalLayer1.OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize * sizeof(float)),
                                                        conv1outputAll,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");

            Console.WriteLine("\nConvLayer1 output activations:");
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                float[] layerOutput = new float[convolutionalLayer1.OutputNeurons.NumberOfUnits];
                Array.Copy(conv1outputAll, m * convolutionalLayer1.OutputNeurons.NumberOfUnits, layerOutput, 0, convolutionalLayer1.OutputNeurons.NumberOfUnits);

                Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                for (int j = 0; j < layerOutput.Length; j++)
                    Console.Write("{0}  ", layerOutput[j]);
                Console.WriteLine();
                Console.ReadKey();
            }
            */
            if (nonlinearityType == "ReLU")
                nonlinearityReLU.FeedForward();
            else if (nonlinearityType == "ELU")
                nonlinearityELU.FeedForward();


            /*
            float[] nonlinOutputAll = new float[nonlinearity.OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        nonlinearity.OutputNeurons.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(convolutionalLayer1.OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize * sizeof(float)),
                                                        nonlinOutputAll,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");

            Console.WriteLine("\nNonlinearity output activations:");
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                float[] layerOutput = new float[nonlinearity.OutputNeurons.NumberOfUnits];
                Array.Copy(nonlinOutputAll, m * nonlinearity.OutputNeurons.NumberOfUnits, layerOutput, 0, nonlinearity.OutputNeurons.NumberOfUnits);

                Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                for (int j = 0; j < layerOutput.Length; j++)
                    Console.Write("{0}  ", layerOutput[j]);
                Console.WriteLine();
                Console.ReadKey();
            }
            */

            convolutionalLayer2.FeedForward();

            /*
            float[] conv2outputAll = new float[convolutionalLayer2.OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        convolutionalLayer2.OutputNeurons.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(convolutionalLayer2.OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize * sizeof(float)),
                                                        conv2outputAll,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "NeuralNetwork.ForwardPass Cl.clEnqueueReadBuffer layerInput");

            Console.WriteLine("\nConvLayer2 output activations:");
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                float[] layerOutput = new float[convolutionalLayer2.OutputNeurons.NumberOfUnits];
                Array.Copy(conv2outputAll, m * convolutionalLayer2.OutputNeurons.NumberOfUnits, layerOutput, 0, convolutionalLayer2.OutputNeurons.NumberOfUnits);

                Console.WriteLine("\n --- Mini-batch item {0} -----", m);
                for (int j = 0; j < layerOutput.Length; j++)
                    Console.Write("{0}  ", layerOutput[j]);
                Console.WriteLine();
                Console.ReadKey();
            }
            */

            // Additionally, cumulate inputs onto outputs

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.SkipForward, 0, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SkipForward, 1, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SkipForward, 2, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SkipForward, 3, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.SkipForward,
                                                            1,
                                                            null,
                                                            globalWorkSizePtr,
                                                            localWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
        }


        public override void BackPropagate()
        {
            // Errors have already been backpropagated to input of first convolutional layer (see method UpdateSpeeds)
            // Now just cumulate the gradients coming from the skip connection

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.SkipBackward, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SkipBackward, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SkipBackward, 2, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SkipBackward, 3, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.SkipBackward,
                                                            1,
                                                            null,
                                                            globalWorkSizePtr,
                                                            localWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

        }


        public override void UpdateSpeeds(double learningRate, double momentumCoefficient, double weightDecayCoefficient)
        {
            // Should include backpropagation to input in all member layers

            convolutionalLayer2.UpdateSpeeds(learningRate, momentumCoefficient, weightDecayCoefficient);
            convolutionalLayer2.BackPropagate();

            if (nonlinearityType == "ReLU")
                nonlinearityReLU.BackPropagate();
            else if (nonlinearityType == "ELU")
                nonlinearityELU.BackPropagate();

            convolutionalLayer1.UpdateSpeeds(learningRate, momentumCoefficient, weightDecayCoefficient);
            convolutionalLayer1.BackPropagate();

        }

        public override void UpdateParameters(double weightMaxNorm)
        {
            convolutionalLayer2.UpdateParameters(weightMaxNorm);
            convolutionalLayer1.UpdateParameters(weightMaxNorm);
        }

        #endregion

    }
}
