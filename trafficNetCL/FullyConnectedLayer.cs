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

        public Neurons Input
        {
            get { return input; }
            set { input = value; }
        }

        public Neurons Output
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
            this.output = new Neurons();

            Console.WriteLine("CONSTRUCTOR: is input instantiated? {0}", input != null);
            //Console.WriteLine("Is output instantiated? {0}", output != null);
        }

        public override void Setup(int InputImageWidth, int InputImageHeight, int InputImageDepth) // in case this is the first layer
        {
            this.input.Setup(InputImageWidth * InputImageHeight * InputImageDepth);

            Console.WriteLine("SETUP: Does input of layer 0 exist INSIDE the setup function? {0}", Input != null);
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
