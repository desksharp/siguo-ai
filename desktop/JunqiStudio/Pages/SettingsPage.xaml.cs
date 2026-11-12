using System.IO;
using System.Windows;
using System.Windows.Controls;
using JunqiStudio.Services;

namespace JunqiStudio.Pages;

public partial class SettingsPage : UserControl, IPage
{
    public SettingsPage()
    {
        InitializeComponent();
        Loaded += (_, _) => OnShown();
    }

    public void OnShown()
    {
        var s = Settings.Current;
        ProjectBox.Text = s.ProjectDir;
        PythonBox.Text = s.Python;
        BenchBox.Text = s.BenchDir;
        PortBox.Text = s.PlayPort.ToString();
        var ver = typeof(SettingsPage).Assembly.GetName().Version;
        var src = Updater.SourceTime();
        var avail = src is { } t && t > Updater.BuildTime.AddSeconds(5);
        Ui.Badge(UpdBadge, UpdBadgeText, src == null ? "找不到源码" : avail ? "有新版本" : "已是最新", src == null ? "grey" : avail ? "amber" : "green");
        UpdInfo.Text = $"当前程序编译于 {Updater.BuildTime:yyyy-MM-dd HH:mm}" + (src is { } s2 ? $"；源码最后修改于 {s2:yyyy-MM-dd HH:mm}" : "");
        UpdBtnText.Text = avail ? "立即更新" : "重新编译";
        About.Text = $"军棋 AI 工作台 {ver?.ToString(3)} · .NET {Environment.Version} · 程序目录 {AppContext.BaseDirectory}";
    }

    void Save_Click(object sender, RoutedEventArgs e)
    {
        var s = Settings.Current;
        var errs = new List<string>();
        if (!Directory.Exists(ProjectBox.Text)) errs.Add("项目目录不存在");
        if (!File.Exists(PythonBox.Text)) errs.Add("找不到 python.exe");
        if (!int.TryParse(PortBox.Text, out var port) || port < 1024 || port > 65535) errs.Add("端口要在 1024–65535 之间");
        if (errs.Count > 0)
        {
            Msg.Foreground = Ui.Res("Red");
            Msg.Text = string.Join("；", errs);
            return;
        }
        s.ProjectDir = ProjectBox.Text.Trim();
        s.Python = PythonBox.Text.Trim();
        s.BenchDir = BenchBox.Text.Trim();
        s.PlayPort = port;
        s.Save();
        Msg.Foreground = Ui.Res("Green");
        Msg.Text = "已保存";
    }

    void Update_Click(object sender, RoutedEventArgs e) => MainWindow.Instance.StartUpdate();

    void OpenProject_Click(object sender, RoutedEventArgs e) => Procs.OpenFolder(Settings.Current.ProjectDir);
    void OpenRuns_Click(object sender, RoutedEventArgs e) => Procs.OpenFolder(Settings.Current.RunsDir);

    void Manual_Click(object sender, RoutedEventArgs e)
    {
        var f = Path.Combine(Settings.Current.ProjectDir, "docs", "WINDOWS.md");
        if (File.Exists(f)) try { System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("notepad.exe", $"\"{f}\"")); } catch { }
    }
}
