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
        private float[] deltas;

        #endregion

        
        #region Properties (public)

        public float[] Delta
        {
            get { return deltas; }
        }

        #endregion


        #region Setup methods (to be called once)

        // Constructor
        public FullyConnectedLayer(int nUnits)
        {
            Console.WriteLine("Adding a fully connected layer with {0} units...", nUnits);
            this.numberOfUnits = nUnits;
            this.biases = new float[nUnits];
            this.deltas = new float[nUnits];
            this.input = new Neurons();
            this.output = new Neurons(nUnits);
        }

        public override void ConnectTo(Layer PreviousLayer)
        {
            this.input.ConnectTo(PreviousLayer.Output);
            this.weights = new float[Output.NumberOfUnits, Input.NumberOfUnits];
        }

        public override void SetAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth) // in case this is the first layer
        {
            this.input = new Neurons(InputImageWidth * InputImageHeight * InputImageDepth);
            this.weights = new float[Output.NumberOfUnits, Input.NumberOfUnits];
        }

        public override void InitializeParameters() // only call after either "SetAsFirstLayer()" or "ConnectTo()"
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(nInputUnits)
            // Initialize biases as normally distributed numbers with mean 0 and std 1

            Random rng = new Random(); //reuse this if you are generating many
            double weightsStdDev = 1 / (Math.Sqrt(this.Input.NumberOfUnits));
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

        public override void ForwardOneCPU()
        {
            float[] unbiasedOutput = Utils.MultiplyMatrixByVector(this.weights, (float[])this.input.Get());
            this.output.Set(unbiasedOutput.Zip(this.biases, (x, y) => x + y).ToArray());
        }

        public override void ForwardGPU()
        {

        }

        public override void BackPropOneCPU()
        {
            this.deltas = Utils.MultiplyMatrixTranspByVector(this.weights, this.nextLayer.Delta);
        }






        

    }
}
