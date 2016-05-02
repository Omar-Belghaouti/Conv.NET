using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace WPFAppFromScratch
{
    class App
    {
        [STAThread]
        static void Main()
        {
            //Create an instance of your window.
            MyWindow _window = new MyWindow();

            //Create an instance of a new Application
            System.Windows.Application _wpfApplication = new System.Windows.Application();

            //Run this Application by passing the window object 
            //as the argument
            _wpfApplication.Run(_window);
        }

    }
}
