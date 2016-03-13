using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class DataSet
    {
        // TO-DO: create custom data structure

        private List<float[]> trafficSigns;
        private List<int> labels;
        private int length;

        public List<float[]> TrafficSigns
        {
            get { return trafficSigns; }
        }
        public List<int> Labels
        {
            get { return labels; }
        }

        public int Length
        {
            get { return length; }
        }

        public DataSet()
        {
            Console.WriteLine("Creating dummy data set...");
        }
    }
}
