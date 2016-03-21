using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace TrafficNetCL
{
    class ConvolutionalLayer : Layer
    {

        #region Fields (private)

        private int filterSize; // F
        private int nFilters; // K
        private int strideLength; // S
        private int zeroPadding; // P

        private float[,] inputAsMatrix; // dimension [inputDepth*filterSize^2 x outputWidth^2]
        private float[,] outputAsMatrix; // dimension [nFilters x outputWidth^2]

        private float[,] weights; // dimension [nFilters x inputDepth*filterSize^2]
        private float[] biases; // dimension [nFilters x 1]

        private float[,] weightsUpdateSpeed; // dimension [nFilters x outputWidth^2]
        private float[] biasesUpdateSpeed; // dimension [nFilters x 1]

        /*
        protected int outputWidth; // = (inputWidth - filterSize + 2*zeroPadding) / (strideLength + 1)
        //private int outputHeight; // assume = outputWidth for now
        protected int outputDepth; // = nFilters
        */


        /* Additional fields, inherited from "Layer" class:
         * 
         * protected Neurons input; 
         * protected Neurons output;
         * 
         * protected Layer nextLayer;
         * protected string layerType;
         */

        #endregion


        #region Properties (public)

        #endregion


        #region Setup methods (to be called once)
        // TO-DO: implement constructor
        public ConvolutionalLayer(int FilterSize, int nOfFilters, int StrideLength, int ZeroPadding)
        {
            Console.WriteLine("Adding a convolutional layer with K = {0} filters of size F = {1}, stride length S = {2} and zero padding P = {3}...",
                filterSize, nOfFilters, strideLength, zeroPadding);

            this.type = "Convolutional";

            this.filterSize = FilterSize;
            this.nFilters = nOfFilters;
            this.strideLength = StrideLength;
            this.zeroPadding = ZeroPadding;
        }

        public override void ConnectTo(Layer PreviousLayer)
        {
            // Setup input
            base.ConnectTo(PreviousLayer);
            if ((PreviousLayer.Type != "Convolutional") & (PreviousLayer.Type != "Pooling"))
                throw new System.InvalidOperationException("Attaching convolutional layer to a non-convolutional or non-pooling layer.");
            
            this.inputWidth = PreviousLayer.OutputWidth;
            this.inputDepth = PreviousLayer.OutputDepth;

            // Setup output
            var tmp = (inputWidth - filterSize + 2*zeroPadding)/(strideLength + 1);
            if (Math.Abs(tmp % 1) > GlobalVar.EPSILON)
                throw new System.ArgumentException("Input width, filter size, zero padding and stride length do not fit well. Check the values!");
            this.outputWidth = (int) tmp;
            this.outputDepth = nFilters;
            this.output = new Neurons( (int) (nFilters * Math.Pow(outputWidth, 2) ) );
            
            // Setup I/O matrices
            this.inputAsMatrix = new float[inputDepth * filterSize * filterSize, (int)Math.Pow(outputWidth, 2) ];
            this.outputAsMatrix = new float[nFilters, (int)Math.Pow(outputWidth, 2) ];
        }

        

        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {
            // Setup input
            this.inputWidth = InputWidth;
            this.inputDepth = InputDepth;
            this.input = new Neurons((int)(InputDepth * Math.Pow(InputWidth, 2)));

            // Setup output
            var tmp = (inputWidth - filterSize + 2 * zeroPadding) / (strideLength + 1);
            if (Math.Abs(tmp % 1) > GlobalVar.EPSILON)
                throw new System.ArgumentException("Input width, filter size, zero padding and stride length do not fit well. Check the values!");
            this.outputWidth = (int) tmp;
            this.outputDepth = nFilters;
            this.output = new Neurons((int)(nFilters * Math.Pow(outputWidth, 2)));

            // Setup I/O matrices
            this.inputAsMatrix = new float[inputDepth * filterSize * filterSize, (int)Math.Pow(outputWidth, 2)];
            this.outputAsMatrix = new float[nFilters, (int)Math.Pow(outputWidth, 2)];
        }

        public override void InitializeParameters()
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(numberOfInputUnits)
            // Initialize biases as small positive numbers, e.g. 0.01

            this.weights = new float[nFilters, inputDepth * filterSize * filterSize];
            this.biases = new float[nFilters];

            Random rng = new Random();
            double weightsStdDev = Math.Sqrt(2.0 / this.input.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < weights.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < weights.GetLength(1); iCol++)
                {
                    uniformRand1 = rng.NextDouble();
                    uniformRand2 = rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale using stdDev

                    weights[iRow, iCol] = (float)tmp;
                }

                //uniformRand1 = rng.NextDouble();
                //uniformRand2 = rng.NextDouble();
                // Use a Box-Muller transform to get a random normal(0,1)
                //tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);

                biases[iRow] = 0.01F;//(float)tmp;

            }

            // Also initialize updates speeds to zero (for momentum)
            this.weightsUpdateSpeed = new float[weights.GetLength(0), weights.GetLength(1)];
            this.biasesUpdateSpeed = new float[biases.GetLength(0)];
        }

        #endregion


        #region Training methods

        public override void ForwardOneCPU()
        {
        }

        public override void ForwardBatchCPU()
        {
        }

        public override void ForwardGPU()
        {
        }

        public override void BackPropOneCPU()
        {

        }

        public override void BackPropBatchCPU()
        {
        }

        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {
        }

        #endregion

    }
}
