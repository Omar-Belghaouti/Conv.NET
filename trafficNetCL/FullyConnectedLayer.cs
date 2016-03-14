using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class FullyConnectedLayer : Layer
    {

        #region Fields (private)

        private int numberOfUnits;

        private float[,] weights;
        private float[] biases;
        
        private new float[] deltaInput;

        #endregion

        
        #region Properties (public)

        public override float[] DeltaInput
        {
            get { return deltaInput; }
        }

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// Constructor of fully connected layer type. Specify number of units as argument.
        /// </summary>
        /// <param name="nUnits"></param>
        public FullyConnectedLayer(int nUnits)
        {
            Console.WriteLine("Adding a fully connected layer with {0} units...", nUnits);
            this.numberOfUnits = nUnits;
            this.biases = new float[nUnits];
            this.input = new Neurons();
            this.output = new Neurons(nUnits);
            this.layerType = "FullyConnected";
        }

        public override void ConnectTo(Layer PreviousLayer)
        {
 	        base.ConnectTo(PreviousLayer);
            this.weights = new float[Output.NumberOfUnits, Input.NumberOfUnits];
            this.deltaInput = new float[Input.NumberOfUnits];
        }

        public override void SetAsFirstLayer(int[] InputDimensions)
        {
            this.input = new Neurons(InputDimensions[0] * InputDimensions[1] * InputDimensions[2]);
            this.weights = new float[this.output.NumberOfUnits, this.input.NumberOfUnits];
        }

        public override void InitializeParameters() // only call after either "SetAsFirstLayer()" or "ConnectTo()"
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(nInputUnits)
            // Initialize biases as normally distributed numbers with mean 0 and std 1

            Random rng = new Random(); //reuse this if you are generating many
            double weightsStdDev = 1 / (Math.Sqrt(this.input.NumberOfUnits));
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < this.weights.GetLength(0); iRow++)
            {
                
                for (int iCol = 0; iCol < this.weights.GetLength(1); iCol++)
                {
                    uniformRand1 = rng.NextDouble();
                    uniformRand2 = rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale

                    weights[iRow, iCol] = (float)tmp;
                }

                uniformRand1 = rng.NextDouble();
                uniformRand2 = rng.NextDouble();
                // Use a Box-Muller transform to get a random normal(0,1)
                tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);

                biases[iRow] = (float)tmp;
                
            }
        }
        #endregion


        #region Training methods

        public override void ForwardOneCPU()
        {
            float[] unbiasedOutput = Utils.MultiplyMatrixByVector(this.weights, (float[])this.input.Get());
            this.output.Set(unbiasedOutput.Zip(this.biases, (x, y) => x + y).ToArray());
        }

        public override void ForwardBatchCPU()
        {

        }

        public override void ForwardGPU()
        {

        }

        public override void BackPropOneCPU()
        {
            this.deltaInput = Utils.MultiplyMatrixTranspByVector(this.weights, this.nextLayer.DeltaInput);
        }

        public override void BackPropOneCPU(float[] costGradient)
        {
            throw new System.InvalidOperationException("You are using a fully connected layer as output layer of the network...\nIs it really what you want to do?");
        }


        public override void BackPropBatchCPU()
        {
          
        }

        public override void UpdateParameters(double learningRate)
        {
            // Update weights
            for (int i = 0; i < this.weights.GetLength(0); i++)
            {
                for (int j = 0; j < this.weights.GetLength(1); j++)
                {
                    this.weights[i, j] += (float) (learningRate * this.input.Get()[j] * this.nextLayer.DeltaInput[i]);
                }
            }

            // Update biases
            this.biases = this.biases.Zip(this.nextLayer.DeltaInput, (x, y) => x + (float)(learningRate * y)).ToArray(); // the LINQ way

        }

        #endregion

    }
}
