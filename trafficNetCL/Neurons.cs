using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    public class Neurons
    {
        #region Neuron class fields (private)

        private int nUnits;
        private float[] unitsActivations;
        private float[] delta;

        #endregion


        #region Neuron class properties (public)


        public int NumberOfUnits 
        {
            get { return nUnits;  }
        }

        public void Set(float[] value)
        {
            this.unitsActivations = value;
        }

        public float[] Get()
        {
            return unitsActivations;
        }

        public virtual float[] Delta
        {
            get { return delta; }
            set { this.delta = value; }
        }

        #endregion


        #region Constructors

        public Neurons() // Generic constructor
        {
        }

        public Neurons(int NumberOfUnits) // Constructor for 1D Neurons instance
        {
            this.nUnits = NumberOfUnits;
            this.unitsActivations = new float[nUnits];
            this.delta = new float[nUnits];
        }

        #endregion



    }
}
