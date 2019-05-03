using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;

namespace ClientUnet
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public static byte[] ReadFully(Stream input)
        {
            byte[] buffer = new byte[16 * 1024];
            using (MemoryStream ms = new MemoryStream())
            {
                int read;
                while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                {
                    ms.Write(buffer, 0, read);
                }
                return ms.ToArray();
            }
        }

        public BitmapImage ToImage(byte[] array)
        {
            using (var ms = new System.IO.MemoryStream(array))
            {
                var image = new BitmapImage();
                image.BeginInit();
                image.CacheOption = BitmapCacheOption.OnLoad; // here
                image.StreamSource = ms;
                image.EndInit();
                return image;
            }
        }


        private string fileName;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void BrowseButton_OnClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog op = new OpenFileDialog();
            op.Title = "Select a picture";

            if (op.ShowDialog() == true)
            {
                fileName = op.FileName;
                ImageViewer1.Source = new BitmapImage(new Uri(fileName));
            }
        }

        private async  void Send_OnClick(object sender, RoutedEventArgs e)
        {

            using (FileStream fsSource = new FileStream(fileName,
                FileMode.Open, FileAccess.Read))
            {
                var imageStream = await UploadAsync("http://localhost:5000/predict", fsSource);
                if (imageStream == null)
                {
                    MessageBox.Show("Error on the server");
                    return;
                }

                using (var memStream = new MemoryStream())
                {

                    await imageStream.CopyToAsync(memStream);
                    memStream.Position = 0;

                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = memStream;
                    bitmap.EndInit();
                    bitmap.Freeze();
                    ImageViewer2.Source = bitmap;
                }
            }
        }

        private async void Send_OnClick2(object sender, RoutedEventArgs e)
        {
            using (var client = new HttpClient())
            {
                var response = client.GetAsync("http://localhost:5000/file1").Result;
                BitmapImage bitmap = new BitmapImage();
                if (response != null)
                {
                    using (var stream = await response.Content.ReadAsStreamAsync())
                    using (var memStream = new MemoryStream())
                    {

                        await stream.CopyToAsync(memStream);
                        memStream.Position = 0;
                        bitmap.BeginInit();
                        bitmap.CacheOption = BitmapCacheOption.OnLoad;
                        bitmap.StreamSource = memStream;
                        bitmap.EndInit();
                        bitmap.Freeze();
                        ImageViewer2.Source = bitmap;
                    }
                }

            }
        }

        private async Task<System.IO.Stream> UploadAsync(string actionUrl, Stream paramFileStream)
        {
            HttpContent fileStreamContent = new StreamContent(paramFileStream);
            using (var client = new HttpClient())
            using (var formData = new MultipartFormDataContent())
            {
                formData.Add(fileStreamContent, "image", "file.png");
                var response = await client.PostAsync(actionUrl, formData);
                if (!response.IsSuccessStatusCode)
                {
                    return null;
                }

                return await response.Content.ReadAsStreamAsync();
            }
        }

        public Task<HttpResponseMessage> UploadAsFormDataContent(string url, byte[] image, string name)
        {
            MultipartFormDataContent form = new MultipartFormDataContent
                {
                    { new ByteArrayContent(image, 0, image.Length), "file", name }
                };

            HttpClient client = new HttpClient();
            return client.PostAsync(url, form);
        }

    }
}
