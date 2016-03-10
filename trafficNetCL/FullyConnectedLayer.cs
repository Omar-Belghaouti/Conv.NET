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

        private Neurons input;
        private Neurons output;

        #endregion

        
        #region Properties (public)

        public override Neurons Input
        {
            get { return input; }
            set { input = value; }
        }

        public override Neurons Output
        {
            get { return output; }
        }
         

        #endregion
        

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
        }

        public override void SetupAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth) // in case this is the first layer
        {
            this.input = new Neurons(InputImageWidth * InputImageHeight * InputImageDepth);

            // also initialize weights and biases
        }

        public override void Setup() // after being connected to another layer
        {
            // TO-DO
        }


        public override void ForwardCPU()
        {
      
        }

        public override void ForwardGPU()
        {

        }



    }
}
