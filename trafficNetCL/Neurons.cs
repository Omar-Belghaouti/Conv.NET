#define OPENCL_ENABLED

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    public class Neurons
    {
        #region Neuron class fields (private)

        private int nUnits;

        private float[] unitsActivations;
        private float[] delta;
        
#if OPENCL_ENABLED

        private Mem activationsGPU;
        private Mem deltaGPU;

#endif
        
        #endregion

        #region Neuron class properties (public)


#if OPENCL_ENABLED

        public Mem ActivationsGPU 
        { 
            get { return this.activationsGPU; }
            set { this.activationsGPU = value; }
        }

        public Mem DeltaGPU 
        { 
            get { return this.deltaGPU; }
            set { this.deltaGPU = value; }
        }
#endif

        public int NumberOfUnits 
        {
            get { return nUnits;  }
        }

        public float[] GetHost()
        {
            return unitsActivations;
        }

        public void SetHost(float[] value)
        {
            this.unitsActivations = value;
        }

        public float[] DeltaHost
        {
            get { return delta; }
            set { this.delta = value; }
        }

        #endregion


        #region Constructors

        public Neurons() // Generic constructor
        {
        }

        /// <summary>
        /// Constructor. Use for CPU.
        /// </summary>
        /// <param name="NumberOfUnits"></param>
        public Neurons(int NumberOfUnits)
        {
            this.nUnits = NumberOfUnits;

#if OPENCL_ENABLED
            int bufferSize = sizeof(float) * NumberOfUnits;
            
            this.activationsGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)bufferSize, out CL.Error);
            this.deltaGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)bufferSize, out CL.Error);
            CL.CheckErr(CL.Error, "Neurons constructor: Cl.CreateBuffer");
#else

            this.unitsActivations = new float[nUnits];
            this.delta = new float[nUnits];
#endif
        
        }
        
        #endregion



    }
}
