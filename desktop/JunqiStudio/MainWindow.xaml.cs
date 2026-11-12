using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using JunqiStudio.Pages;
using JunqiStudio.Services;

namespace JunqiStudio;

public partial class MainWindow : Window
{
    public static MainWindow Instance { get; private set; } = null!;

    readonly DashboardPage _dashboard = new();
    readonly TrainingPage _training = new();
    readonly PlayPage _play = new();
    readonly DataPage _data = new();
    readonly SettingsPage _settings = new();

    public MainWindow()
    {
        Instance = this;
        InitializeComponent();
        Host.Content = _dashboard;
        Hub.Updated += UpdateGpu;
        var updTimer = new System.Windows.Threading.DispatcherTimer { Interval = TimeSpan.FromSeconds(60) };
        updTimer.Tick += (_, _) => CheckUpdate();
        updTimer.Start();
        CheckUpdate();
        Loaded += async (_, _) =>
        {
            await Gpu.DetectTorch();
            UpdateGpu();
            if (App.ShotDir != null) await TakeShots(App.ShotDir);
        };
    }

    void Nav_Checked(object sender, RoutedEventArgs e)
    {
        if (Host == null) return;
        Host.Content = sender switch
        {
            _ when sender == NavTraining => _training,
            _ when sender == NavPlay => _play,
            _ when sender == NavData => _data,
            _ when sender == NavSettings => _settings,
            _ => _dashboard,
        };
        (Host.Content as IPage)?.OnShown();
    }

    public void ShowTraining(string? runName = null)
    {
        _training.Select(runName);
        NavTraining.IsChecked = true;
    }

    public void NewRun()
    {
        NavTraining.IsChecked = true;
        _training.NewRun();
    }

    public void ShowPlay() => NavPlay.IsChecked = true;
    public void ShowData() => NavData.IsChecked = true;

    public void CheckUpdate()
    {
        var src = Updater.SourceTime();
        var avail = src is { } t && t > Updater.BuildTime.AddSeconds(5);
        UpdatePill.Visibility = avail && App.ShotDir == null ? Visibility.Visible : Visibility.Collapsed;
        if (avail) UpdatePillText.Text = $"有新版本 {src:MM/dd HH:mm} · 点此更新";
    }

    void UpdatePill_Click(object sender, System.Windows.Input.MouseButtonEventArgs e) => StartUpdate();

    /// <summary>重新编译工作台；成功后由用户点“重启并更新”换上新版本。</summary>
    public async void StartUpdate()
    {
        if (Updater.DotnetPath() == null)
        {
            MessageBox.Show(this, "没有找到 .NET SDK（dotnet.exe），无法编译新版本。", "更新程序", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        var w = new OutputWindow("更新程序", $"从 {Updater.SourceDir} 重新编译工作台（约 20–40 秒）");
        if (await Updater.Build(w))
            w.ShowAction("重启并更新", Updater.SwapAndRestart);
    }

    void UpdateGpu()
    {
        GpuNameText.Text = string.IsNullOrEmpty(Gpu.Name) ? "未检测到显卡" : Gpu.Name.Replace("AMD ", "");
        GpuDot.Fill = (Brush)FindResource(Gpu.TorchOk == true ? "Green" : Gpu.TorchOk == false ? "Red" : "Grey");
        var mem = Gpu.TotalGB > 0 ? $" · {Hub.GpuUsedGB:0.0}/{Gpu.TotalGB:0} GB" : "";
        GpuLoadText.Text = Gpu.TorchOk == false ? "PyTorch 无法使用显卡" : $"占用 {Hub.GpuUtil:0}%{mem}";
    }

    async Task TakeShots(string dir)
    {
        Directory.CreateDirectory(dir);
        Width = 1320; Height = 860;
        var pages = new (RadioButton nav, string name)[]
            { (NavDashboard, "1_dashboard"), (NavTraining, "2_training"), (NavPlay, "3_play"), (NavData, "4_data"), (NavSettings, "5_settings") };
        await Task.Delay(4000);
        bool startedPlay = false;
        foreach (var (nav, name) in pages)
        {
            if (nav == NavPlay && Hub.Play == null)
            {
                PlayService.Start(PlayService.DefaultModel());
                startedPlay = true;
            }
            nav.IsChecked = true;
            await Task.Delay(nav == NavPlay ? 9000 : 2500);
            Capture(Path.Combine(dir, name + ".png"));
        }
        if (startedPlay && Hub.Play is { } p) PlayService.Stop(p);
        _training.ShowNewRunDialogForShot(Path.Combine(dir, "6_newrun.png"));
        await Task.Delay(2500);
        _training.ShowNewRunDialogForShot(Path.Combine(dir, "7_newrun_rl.png"), rl: true);
        await Task.Delay(2500);
        Application.Current.Shutdown();
    }

    public static void SaveVisual(FrameworkElement el, string file)
    {
        var dpi = VisualTreeHelper.GetDpi(el);
        var bmp = new RenderTargetBitmap((int)(el.ActualWidth * dpi.DpiScaleX), (int)(el.ActualHeight * dpi.DpiScaleY),
                                         dpi.PixelsPerInchX, dpi.PixelsPerInchY, PixelFormats.Pbgra32);
        var dv = new DrawingVisual();
        using (var dc = dv.RenderOpen())
            dc.DrawRectangle(new VisualBrush(el), null, new Rect(0, 0, el.ActualWidth, el.ActualHeight));
        bmp.Render(dv);
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        using var fs = File.Create(file);
        enc.Save(fs);
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")] static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
    [System.Runtime.InteropServices.DllImport("user32.dll")] static extern IntPtr GetDC(IntPtr hwnd);
    [System.Runtime.InteropServices.DllImport("user32.dll")] static extern int ReleaseDC(IntPtr hwnd, IntPtr hdc);
    [System.Runtime.InteropServices.DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleDC(IntPtr hdc);
    [System.Runtime.InteropServices.DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int w, int h);
    [System.Runtime.InteropServices.DllImport("gdi32.dll")] static extern IntPtr SelectObject(IntPtr hdc, IntPtr obj);
    [System.Runtime.InteropServices.DllImport("gdi32.dll")] static extern bool DeleteObject(IntPtr obj);
    [System.Runtime.InteropServices.DllImport("gdi32.dll")] static extern bool DeleteDC(IntPtr hdc);

    /// <summary>用 PrintWindow(PW_RENDERFULLCONTENT) 抓整个窗口，内嵌的 WebView2 也能抓到。</summary>
    void Capture(string file)
    {
        var hwnd = new System.Windows.Interop.WindowInteropHelper(this).Handle;
        var dpi = VisualTreeHelper.GetDpi(this);
        int w = (int)(ActualWidth * dpi.DpiScaleX), h = (int)(ActualHeight * dpi.DpiScaleY);
        var screen = GetDC(IntPtr.Zero);
        var mem = CreateCompatibleDC(screen);
        var bmp = CreateCompatibleBitmap(screen, w, h);
        var old = SelectObject(mem, bmp);
        PrintWindow(hwnd, mem, 2);
        SelectObject(mem, old);
        var src = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(bmp, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
        DeleteObject(bmp); DeleteDC(mem); ReleaseDC(IntPtr.Zero, screen);
        var enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(src));
        using var fs = File.Create(file);
        enc.Save(fs);
    }
}

public interface IPage
{
    void OnShown();
}
