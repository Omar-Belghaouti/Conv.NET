using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    [Serializable]
    class SoftMax : Layer
    {
        #region Fields

        private List<double[]> outputClassScores;

        #endregion


        #region Properties

        public List<double[]> OutputClassScores
        {
            get { return outputClassScores; }
        }

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of Softmax layer.
        /// </summary>
        /// <param name="Beta"></param>
        public SoftMax()
        {
            this.type = "SoftMax";
            this.outputClassScores = new List<double[]>();
        }


        public override void SetupOutput()
        {
            this.nOutputUnits = nInputUnits;
            this.outputNeurons = new Neurons(nOutputUnits);
        }

        public void SetupOutputScores(int miniBatchSize)
        {
            for (int m = 0; m < miniBatchSize; m++)
                outputClassScores.Add(new double[nOutputUnits]);
        }

        #endregion


        #region Methods

        public override void FeedForward()
        {
#if TIMING_LAYERS
            Utils.SoftmaxTimer.Start();
#endif

            int nActivations = nInputUnits*inputNeurons.MiniBatchSize;

            // get preactivations
#if OPENCL_ENABLED
            float[] preActivations = new float[nActivations];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                        inputNeurons.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nActivations),
                                                        preActivations,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "SoftMax.FeedForward(): clEnqueueReadBuffer preActivations");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            double[] preActivations = new double[nActivations];

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                for (int i = 0; i < nInputUnits; i++)
                {
                    preActivations[m*nInputUnits + i] = inputNeurons.GetHost()[m][i];
                }
            }
#endif

            
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                // rescale to improve numerical stability
                double maxPreactivation = preActivations[m];
                for (int i = 1; i < nInputUnits; i++)
                {
                    if (preActivations[m*nInputUnits + i] > maxPreactivation)
                        maxPreactivation = preActivations[m*nInputUnits + i];
                }

                double[] tmpActivations = new double[nOutputUnits];
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpActivations[i] = Math.Exp(preActivations[m*nInputUnits + i] - maxPreactivation);
                }
                double sum = tmpActivations.Sum();
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpActivations[i] /= sum;
                    //if (tmpActivations[i] < 0)
                        //System.Diagnostics.Debugger.Launch();
                }
                outputClassScores[m] = tmpActivations;
            }

#if TIMING_LAYERS
            Utils.SoftmaxTimer.Stop();
#endif
        }


        public override void BackPropagate()
        {
            throw new System.NotImplementedException("Called BackPropagate() method of SoftMax layer. Don't do it! Just feed the gradient back to the previous layer!");
            // NO backprop here!!
            // Compute directly input.Delta from cross-entropy cost: faster and numerically more stable
        }

        #endregion


    }
}
