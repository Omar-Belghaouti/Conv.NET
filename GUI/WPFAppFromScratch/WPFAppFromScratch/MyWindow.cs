using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

//Add these namespaces
using System.Windows;
using System.Windows.Controls;

using O

namespace WPFAppFromScratch
{
    //This will be your custom window class which is derieved    
    //from the base class Window.
    class MyWindow : Window
    {
        //Declare some UI controls to be placed inside the
        //window.
        Button _searchButton;
        TextBox _searchTextBox;

        //The controls can be placed only inside a panel.
        StackPanel _panel;

        public MyWindow()
        {
            //This is created just to show a reference , the 
            //below code can aswell be witten wihin this     
            //constructor.
            InitializeComponent();
        }

        void InitializeComponent()
        {
            _searchButton = new Button { Height = 30, Width = 100, Content = "Search" };
            _searchTextBox = new TextBox { Height = 30, Width = 100 };

            _panel = new StackPanel();

            //Add the controls inside the panel.
            _panel.Children.Add(_searchButton);
            _panel.Children.Add(_searchTextBox);

            //Set this panel as the content for this window.
            this.Content = _panel;
        }
    }
}
