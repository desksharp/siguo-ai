using System.Windows;
using JunqiStudio.Services;

namespace JunqiStudio;

public partial class App : Application
{
    /// <summary>--shot &lt;目录&gt;：依次打开各页面截图后退出（远程检查界面用）。</summary>
    public static string? ShotDir { get; private set; }

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        for (int i = 0; i < e.Args.Length - 1; i++)
            if (e.Args[i] == "--shot") ShotDir = e.Args[i + 1];
        DispatcherUnhandledException += (_, ex) =>
        {
            MessageBox.Show(ex.Exception.Message, "出错了", MessageBoxButton.OK, MessageBoxImage.Warning);
            ex.Handled = true;
        };
        Hub.Start();
        new MainWindow().Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Hub.Stop();
        base.OnExit(e);
    }
}
