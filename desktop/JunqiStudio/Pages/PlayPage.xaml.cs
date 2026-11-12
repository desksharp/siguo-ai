using System.IO;
using System.Net.Http;
using System.Windows;
using System.Windows.Controls;
using JunqiStudio.Services;

namespace JunqiStudio.Pages;

public sealed record ModelChoice(string Path, string Label)
{
    public override string ToString() => Label;
}

public partial class PlayPage : UserControl, IPage
{
    bool _webReady, _busy;
    string _shownUrl = "";
    static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(1) };

    public PlayPage()
    {
        InitializeComponent();
        Hub.Updated += () => { if (IsVisible) Refresh(); };
        Loaded += (_, _) => { LoadModels(); Refresh(); };
    }

    public void OnShown()
    {
        LoadModels();
        Refresh();
    }

    void LoadModels()
    {
        var runs = RunInfo.LoadAll().Where(r => r.HasModel).ToList();
        var items = runs.Select(r => new ModelChoice(r.ModelPath,
            $"{r.Name} · {r.SizeText} · " + (r.HasSummary ? $"对老师 {r.VsTeacher?.Score ?? "-"}" : $"第 {r.LastEvalStep:N0} 步"))).ToList();
        var current = Hub.Play is { } p ? Procs.Norm(p.Arg, Settings.Current.ProjectDir) : Procs.Norm(PlayService.DefaultModel(), Settings.Current.ProjectDir);
        ModelBox.ItemsSource = items;
        ModelBox.SelectedItem = items.FirstOrDefault(i => Procs.Norm(i.Path, Settings.Current.ProjectDir) == current) ?? items.FirstOrDefault();
    }

    async void Refresh()
    {
        var p = Hub.Play;
        StopBtn.IsEnabled = p != null;
        if (p != null)
        {
            Ui.Badge(Badge, BadgeText, "运行中", "green");
            StartText.Text = "换模型重启";
            StartIcon.Text = "";
            var url = PlayService.Url;
            if (_shownUrl != url && !_busy)
            {
                _busy = true;
                try { if (await Alive(url)) await Show(url); } finally { _busy = false; }
            }
        }
        else
        {
            Ui.Badge(Badge, BadgeText, "已停止", "grey");
            StartText.Text = "启动";
            StartIcon.Text = "";
            _shownUrl = "";
            Web.Visibility = Visibility.Collapsed;
            Placeholder.Visibility = Visibility.Visible;
            PlaceholderText.Text = "对弈服务还没启动";
        }
    }

    static async Task<bool> Alive(string url)
    {
        try { using var r = await Http.GetAsync(url + "/api/board"); return r.IsSuccessStatusCode; } catch { return false; }
    }

    async Task Show(string url)
    {
        try
        {
            if (!_webReady)
            {
                var data = Path.Combine(AppContext.BaseDirectory, "webview2");
                var env = await Microsoft.Web.WebView2.Core.CoreWebView2Environment.CreateAsync(null, data);
                await Web.EnsureCoreWebView2Async(env);
                _webReady = true;
            }
            _shownUrl = url;
            Web.Source = new Uri(url);
            Web.Visibility = Visibility.Visible;
            Placeholder.Visibility = Visibility.Collapsed;
        }
        catch (Exception ex)
        {
            PlaceholderText.Text = "内嵌浏览器（WebView2）无法启动：" + ex.Message;
        }
    }

    async void Start_Click(object sender, RoutedEventArgs e)
    {
        var model = (ModelBox.SelectedItem as ModelChoice)?.Path ?? "";
        if (Hub.Play is { } p)
        {
            PlayService.Stop(p);
            await Task.Delay(800);
        }
        _shownUrl = "";
        Web.Visibility = Visibility.Collapsed;
        Placeholder.Visibility = Visibility.Visible;
        PlaceholderText.Text = "正在启动…";
        PlayService.Start(model);
        for (int i = 0; i < 30 && !await Alive(PlayService.Url); i++) await Task.Delay(300);
        await Hub.Refresh();
    }

    async void Stop_Click(object sender, RoutedEventArgs e)
    {
        if (Hub.Play is { } p) PlayService.Stop(p);
        await Task.Delay(500);
        await Hub.Refresh();
    }

    void Reload_Click(object sender, RoutedEventArgs e) { if (_webReady && Web.Visibility == Visibility.Visible) Web.Reload(); }
    void Browser_Click(object sender, RoutedEventArgs e) => Procs.OpenUrl(PlayService.Url);
}
