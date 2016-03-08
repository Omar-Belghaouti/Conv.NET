using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace trafficNetCL
{
    interface ILayer
    {
        void ForwardOne();

        void ForwardBatch();

        void BackProp();
    }
}
