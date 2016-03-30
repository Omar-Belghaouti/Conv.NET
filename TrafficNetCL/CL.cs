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

        public static ErrorCode Error;
        public static Event Event;

        #endregion

        #region CL helper class properties (public)

        /*
        public static ErrorCode Error
        { 
            get {return Error; }
            set { Error = value; }
        }
         */

        public static Context Context
        {
            get { return _context; }
            set { _context = value; }
        }

        public static CommandQueue Queue
        {
            get { return _queue; }
            set { _queue = value; }
        }

        #endregion


        #region Kernels (public)

        public static Kernel FCForward;


        #endregion

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
            Console.WriteLine("\nUsing device {0}\n", deviceNames[deviceID]);

            /*
            if (Cl.GetDeviceInfo(_device, DeviceInfo.ImageSupport, out Error).CastTo<Bool>() == Bool.False)
            {
                Console.WriteLine("No image support.");
                return;
            }
            */

            // Create OpenCL context
            _context = Cl.CreateContext(null, 1, new[] { _device }, ContextNotify, IntPtr.Zero, out Error);    //Second parameter is amount of devices
            CheckErr(Error, "CL.Setup: Cl.CreateContext");
        }


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
            FCForward = LoadBuildKernel(kernelsPath, "FCForward");
        }



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

    }
}
