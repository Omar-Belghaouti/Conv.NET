using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{

    class SoftMax : Layer
    {
        #region Fields

        private List<float[]> outputClassScores;

        #endregion


        #region Properties

        public override List<float[]> OutputClassScores
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
            this.outputClassScores = new List<float[]>();
        }

        /// <summary>
        ///  Connect current layer to layer given as argument.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);

            this.nOutputUnits = PreviousLayer.OutputNeurons.NumberOfUnits;
            this.outputNeurons = new Neurons(this.nOutputUnits);

        }

        #endregion


        #region Methods

        public override void FeedForward()
        {
            // workaround to create list of output scores once in the beginning
            if (outputClassScores.Count == 0)
            {
                for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
                outputClassScores.Add( new float[nOutputUnits] );
            }

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                // get preactivations
                float[] preActivations = new float[nInputUnits];
#if OPENCL_ENABLED
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                            inputNeurons.ActivationsGPU[m], // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(sizeof(float) * nInputUnits),
                                                            preActivations,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "SoftMax.FeedForward(): clEnqueueReadBuffer preActivations");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                preActivations = inputNeurons.GetHost()[m];
#endif

                // rescale to improve numerical stability
                float maxPreactivation = preActivations[0];
                for (int i = 1; i < nInputUnits; i++)
                {
                    if (preActivations[i] > maxPreactivation)
                        maxPreactivation = preActivations[i];
                }

                float[] tmpActivations = new float[nOutputUnits];
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpActivations[i] = (float)Math.Exp(preActivations[i] - maxPreactivation);
                }
                float sum = tmpActivations.Sum();
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpActivations[i] /= sum;
                }

                outputClassScores[m] = tmpActivations;
            }
        }


        public override void BackPropagate()
        {
            throw new System.InvalidOperationException("Called BackPropagate() method of SoftMax layer. Don't do it! Just feed the gradient back to the previous layer!");
            // NO backprop here!!
            // Compute directly input.Delta from cross-entropy cost: faster and numerically more stable
        }

        #endregion


    }
}
