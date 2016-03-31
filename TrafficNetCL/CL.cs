using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    static class CL
    {

        #region CL helper class fields

        private static Context _context;
        private static Device _device;
        private static CommandQueue _queue;

        public static List<int> maxWorkItemSizes;
        public static int maxWorkGroupSize;
        public static int maxComputeUnits;

        public static ErrorCode Error;
        public static Event Event;

        #endregion


        #region CL helper class properties (public)

        public static Context Context
        {
            get { return _context; }
            set { _context = value; }
        }

        public static Device Device
        {
            get { return _device; }
        }

        public static CommandQueue Queue
        {
            get { return _queue; }
            set { _queue = value; }
        }

        #endregion


        #region Kernels (public)

        // FullyConnected layer
        public static Kernel FCForward;
        public static Kernel FCBackward;
        public static Kernel FCUpdateParameters;

        // ReLU layer
        public static Kernel ReLUForward;
        public static Kernel ReLUBackward;

        // Softmax layer
        public static Kernel SoftmaxForward;

        #endregion


        #region OpenCL setup and finalization

        public static void Setup()
        {
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    OpenCL setup");
            Console.WriteLine("=========================================\n");
            
            int deviceID; // will be asked to user

            List<Device> devicesList = new List<Device>();
            List<string> deviceNames = new List<string>();
            List<string> platformNames = new List<string>();
            int nDevices = 0;

            // Get list of available platforms
            Console.WriteLine("\nSearching for OpenCL-capable platforms... ");
            Platform[] platforms = Cl.GetPlatformIDs(out Error);
            CheckErr(Error, "CL.Setup: Cl.GetPlatformIDs");
            Console.WriteLine("{0} platforms found.\n", platforms.Length);

            //Console.WriteLine("\n=============================================\n");
            foreach (Platform platform in platforms)
            {
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out Error).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(Error, "CL.Setup: Cl.GetPlatformInfo");

                // Get all available devices for this platform and list them on screen
                foreach (Device device in Cl.GetDeviceIDs(platform, DeviceType.All, out Error))
                {
                    CheckErr(Error, "CL.Setup: Cl.GetDeviceIDs");
                    string deviceName = Cl.GetDeviceInfo(device, DeviceInfo.Name, out Error).ToString();
                    CheckErr(Error, "CL.Setup: Cl.GetDeviceInfo");
                    Console.WriteLine("Device {0}: {1}", nDevices, deviceName);

                    devicesList.Add(device);
                    deviceNames.Add(deviceName);
                    platformNames.Add(platformName);
                    nDevices++;
                }
                Console.WriteLine();
            }

            if (nDevices == 0)
            {
                Console.WriteLine("FATAL ERROR: No devices found!");
                return;
            }

            Console.Write("Enter ID of device to use: ");
            deviceID = int.Parse(Console.ReadLine()); 

            // Select device according to user's choice
            _device = devicesList[deviceID];
            Console.WriteLine("\nUsing device {0}", deviceNames[deviceID]);

            // Create OpenCL context
            _context = Cl.CreateContext(null, 1, new[] { _device }, ContextNotify, IntPtr.Zero, out Error);    //Second parameter is amount of devices
            CheckErr(Error, "CL.Setup: Cl.CreateContext");

            // Extract some device info
            maxWorkItemSizes = Cl.GetDeviceInfo(_device, DeviceInfo.MaxWorkItemSizes, out Error).CastToEnumerable<int>(new int[] { 0, 1, 2 }).ToList();
            Console.WriteLine("Max work item sizes: {0}, {1}, {2}", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            maxWorkGroupSize = Cl.GetDeviceInfo(_device, DeviceInfo.MaxWorkGroupSize, out Error).CastTo<int>();
            Console.WriteLine("Max work group size: {0}", maxWorkGroupSize);

            maxComputeUnits = Cl.GetDeviceInfo(_device, DeviceInfo.MaxComputeUnits, out Error).CastTo<int>();
            Console.WriteLine("Max compute units: {0}", maxComputeUnits);

        }


        public static void Finalize() // bad practice?
        {
            Cl.ReleaseCommandQueue(_queue);
        }


        #endregion


        #region Kernel loading and building

        public static Kernel LoadBuildKernel(string kernelFilePath, string kernelName)
        {

            // Attempt to read file
            if (!System.IO.File.Exists(kernelFilePath))
            {
                Console.WriteLine("Program doesn't exist at path " + kernelFilePath);
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            string kernelSource = System.IO.File.ReadAllText(kernelFilePath);

            // Create program
            OpenCL.Net.Program clProgram = Cl.CreateProgramWithSource(_context, 1, new[] { kernelSource }, null, out Error);
            CheckErr(Error, "CL.LoadAndBuildKernel: Cl.CreateProgramWithSource");

            //Compile kernel source
            Error = Cl.BuildProgram(clProgram, 1, new[] { _device }, string.Empty, null, IntPtr.Zero);
            CheckErr(Error, "CL.LoadAndBuildKernel: Cl.BuildProgram");

            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(clProgram, _device, ProgramBuildInfo.Status, out Error).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(Error, "CL.LoadAndBuildKernel: Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(Cl.GetProgramBuildInfo(clProgram, _device, ProgramBuildInfo.Log, out Error));
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            //Create the required kernel (entry function)
            Kernel kernel = Cl.CreateKernel(clProgram, kernelName, out Error);
            CheckErr(Error, "CL.LoadAndBuildKernel: Cl.CreateKernel");

            return kernel;
        }


        public static void LoadKernels(string kernelsPath)
        {
            // FullyConnected layer

            string fcForwardName = "FCForward";
            FCForward = LoadBuildKernel(kernelsPath + "/" + fcForwardName + ".cl", fcForwardName);

            string fcBackwardName = "FCBackward";
            FCBackward = LoadBuildKernel(kernelsPath + "/" + fcBackwardName + ".cl", fcBackwardName);

            string fcUpdateParametersName = "FCUpdateParameters";
            FCUpdateParameters = LoadBuildKernel(kernelsPath + "/" + fcUpdateParametersName + ".cl", fcUpdateParametersName);

            // ReLU layer

            string reluForwardName = "ReLUForward";
            ReLUForward = LoadBuildKernel(kernelsPath + "/" + reluForwardName + ".cl", reluForwardName);

            string reluBackwardName = "FCBackward";
            ReLUBackward = LoadBuildKernel(kernelsPath + "/" + reluBackwardName + ".cl", reluBackwardName);

            // Softmax layer

            string softmaxForwardName = "SoftmaxForward";
            SoftmaxForward = LoadBuildKernel(kernelsPath + "/" + softmaxForwardName + ".cl", softmaxForwardName);

        }

        #endregion


        #region Helper methods

        public static void CheckErr(ErrorCode err, string name)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
            }
        }

        public static void ContextNotify(string errInfo, byte[] data, IntPtr cb,
            IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        #endregion

    }
}
