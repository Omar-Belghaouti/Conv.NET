﻿using System;
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

        private List<double[]> outputClassScores;

        #endregion


        #region Properties

        public override List<double[]> OutputClassScores
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

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                // get preactivations
#if OPENCL_ENABLED
                float[] preActivations = new float[nInputUnits];
                OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                            inputNeurons.ActivationsGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(sizeof(float) * nInputUnits),
                                                            preActivations,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "SoftMax.FeedForward(): clEnqueueReadBuffer preActivations");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                double[] preActivations = inputNeurons.GetHost()[m];
#endif

                // rescale to improve numerical stability
                double maxPreactivation = preActivations[0];
                for (int i = 1; i < nInputUnits; i++)
                {
                    if (preActivations[i] > maxPreactivation)
                        maxPreactivation = preActivations[i];
                }

                double[] tmpActivations = new double[nOutputUnits];
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpActivations[i] = Math.Exp(preActivations[i] - maxPreactivation);
                }
                double sum = tmpActivations.Sum();
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpActivations[i] /= sum;
                    if (tmpActivations[i] < 0)
                        System.Diagnostics.Debugger.Launch();
                }

                outputClassScores[m] = tmpActivations;
            }
#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif
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
