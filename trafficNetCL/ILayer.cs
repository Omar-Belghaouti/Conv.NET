using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace trafficNetCL
{
    interface ILayer
    {
        void InitializeWeightsAndBiases();

        void ForwardOne();

        void ForwardBatch();

        void BackProp();

        void UpdateWeights();

        // Properties
        float[, ,] Input { get; set; }
        float[, ,] Output { get; set; }
    }
}
