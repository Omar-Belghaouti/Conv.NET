using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    [Serializable]
    public class Neurons
    {
        // CLEAN 

        #region Neuron class fields (private)

        private int nUnits;
        private int miniBatchSize;

#if OPENCL_ENABLED
        [NonSerialized]
        private Mem activationsGPU;

        [NonSerialized]
        private Mem deltaGPU;
#else
        private List<double[]> activations;
        private List<double[]> delta;
#endif

        #endregion


        #region Neuron class properties (public)

        public int NumberOfUnits
        {
            get { return nUnits; }
        }

        public int MiniBatchSize
        {
            get { return miniBatchSize; }
            set { this.miniBatchSize = value; }
        }

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
#else       
        public List<double[]> GetHost()
        {
            return activations;
        }

        public void SetHost(int iExample, double[] value)
        {
            this.activations[iExample] = value;
        }

        public List<double[]> DeltaHost
        {
            get { return delta; }
            set { this.delta = value; }
        }

#endif

        #endregion


        #region Constructors

        /// <summary>
        /// Neurons constructor.
        /// </summary>
        /// <param name="NumberOfUnits"></param>
        public Neurons(int NumberOfUnits)
        {
            this.nUnits = NumberOfUnits;

#if OPENCL_ENABLED
            /*
            this.activationsGPU = new List<Mem>();
            this.activationsGPU.Add( (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * NumberOfUnits),
                                                            out OpenCLSpace.ClError) );
            

            this.deltaGPU = new List<Mem>();
            
            this.deltaGPU.Add( (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)(sizeof(float) * NumberOfUnits),
                                                    out OpenCLSpace.ClError) );

            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Neurons constructor: Cl.CreateBuffer");
            */
#else
            this.activations = new List<double[]>();
            this.activations.Add(new double[nUnits]);

            this.delta = new List<double[]>();
            this.delta.Add(new double[nUnits]);
#endif

        }
        
        #endregion

        public void SetupBuffers(int MiniBatchSize)
        {
            this.miniBatchSize = MiniBatchSize;

            
#if OPENCL_ENABLED
            this.activationsGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * NumberOfUnits * MiniBatchSize),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer Neurons.activationsGPU");
            OpenCLSpace.WipeBuffer(activationsGPU, NumberOfUnits * MiniBatchSize, typeof(float));

            this.deltaGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)(sizeof(float) * NumberOfUnits * MiniBatchSize),
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.WipeBuffer(activationsGPU, NumberOfUnits * MiniBatchSize, typeof(float));
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer Neurons.deltaGPU");
            
#else
            for (int m = 0; m < MiniBatchSize; m++)
            {
                this.activations.Add(new double[nUnits]);
                this.delta.Add(new double[nUnits]);
            }
#endif


        }



    }
}
