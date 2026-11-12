using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using JunqiStudio.Services;
using ScottPlot;
using ScottPlot.WPF;

namespace JunqiStudio.Pages;

/// <summary>左侧列表里的一项。</summary>
public sealed class RunItem : INotifyPropertyChanged
{
    public string Name { get; init; } = "";
    string _status = "", _line2 = "", _line3 = "";
    double _progress;
    Brush _bg = Brushes.Transparent, _fg = Brushes.Black;
    public string Status { get => _status; set => Set(ref _status, value); }
    public string Line2 { get => _line2; set => Set(ref _line2, value); }
    public string Line3 { get => _line3; set => Set(ref _line3, value); }
    public double Progress { get => _progress; set => Set(ref _progress, value); }
    public Brush StatusBg { get => _bg; set => Set(ref _bg, value); }
    public Brush StatusFg { get => _fg; set => Set(ref _fg, value); }

    public event PropertyChangedEventHandler? PropertyChanged;
    void Set<T>(ref T f, T v, [CallerMemberName] string? n = null)
    {
        if (EqualityComparer<T>.Default.Equals(f, v)) return;
        f = v;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(n));
    }
}

public partial class TrainingPage : UserControl, IPage
{
    readonly ObservableCollection<RunItem> _items = new();
    readonly Dictionary<string, RunInfo> _runs = new();
    string? _selected, _pendingSelect;
    string _plotKey = "", _consoleKey = "";

    static readonly ScottPlot.Color CTrain = ScottPlot.Color.FromHex("#5B4FE9");
    static readonly ScottPlot.Color CEval = ScottPlot.Color.FromHex("#F59E0B");
    static readonly ScottPlot.Color CLight = ScottPlot.Color.FromHex("#CFCBF8");

    sealed record Series(string Label, ScottPlot.Color Color, float Width, float Marker, List<(double X, double Y)> Pts);

    public TrainingPage()
    {
        InitializeComponent();
        RunList.ItemsSource = _items;
        foreach (var wp in new[] { PlotLoss, PlotPolicy, PlotBelief, PlotValue })
        {
            StylePlot(wp);
            // 图表只看不动：关掉缩放/拖动，滚轮交给页面滚动
            wp.UserInputProcessor.Disable();
            wp.PreviewMouseWheel += (_, e) =>
            {
                e.Handled = true;
                Detail.ScrollToVerticalOffset(Detail.VerticalOffset - e.Delta);
            };
        }
        Hub.Updated += () => { if (IsVisible) Refresh(); };
        Loaded += (_, _) => Refresh();
    }

    public void OnShown() => Refresh();

    public void Select(string? name)
    {
        if (name == null) return;
        _pendingSelect = name;
        Refresh();
    }

    void Refresh()
    {
        var runs = RunInfo.LoadAll();
        _runs.Clear();
        foreach (var r in runs) _runs[r.Name] = r;

        // 列表：按名字原地更新，避免每 2 秒重建导致闪烁
        var names = runs.Select(r => r.Name).ToList();
        foreach (var it in _items.Where(i => !names.Contains(i.Name)).ToList()) _items.Remove(it);
        for (int i = 0; i < runs.Count; i++)
        {
            var r = runs[i];
            var it = _items.FirstOrDefault(x => x.Name == r.Name);
            if (it == null) { it = new RunItem { Name = r.Name }; _items.Insert(Math.Min(i, _items.Count), it); }
            else if (_items.IndexOf(it) != i) _items.Move(_items.IndexOf(it), Math.Min(i, _items.Count - 1));
            Fill(it, r);
        }
        EmptyText.Visibility = runs.Count == 0 ? Visibility.Visible : Visibility.Collapsed;

        if (_pendingSelect != null && _runs.ContainsKey(_pendingSelect))
        {
            _selected = _pendingSelect;
            _pendingSelect = null;
        }
        _selected ??= runs.FirstOrDefault(r => Hub.TrainingOf(r.Dir) != null || r.RemoteRunning)?.Name ?? runs.FirstOrDefault()?.Name;
        var sel = _items.FirstOrDefault(i => i.Name == _selected);
        if (RunList.SelectedItem != sel) RunList.SelectedItem = sel;
        ShowDetail(_selected != null && _runs.TryGetValue(_selected, out var run) ? run : null);
    }

    static (string text, string kind) StatusOf(RunInfo r)
    {
        var task = Hub.TrainingOf(r.Dir);
        if (task != null) return r.CurrentStep >= r.Steps && r.Steps > 0 ? ("评估中", "amber") : ("运行中", "green");
        if (r.RemoteRunning) return r.CurrentStep >= r.Steps && r.Steps > 0 ? ("远程评估中", "amber") : ("远程运行中", "green");
        if (r.HasSummary) return ("已完成", "primary");
        if (r.RemoteHost.Length > 0) return ("远程无响应", "amber");
        return r.CurrentStep > 0 ? ("已中断", "amber") : ("未开始", "grey");
    }

    static void Fill(RunItem it, RunInfo r)
    {
        var (text, kind) = StatusOf(r);
        it.Status = text;
        (it.StatusBg, it.StatusFg) = kind switch
        {
            "green" => (Ui.Res("GreenSoft"), Ui.Res("Green")),
            "amber" => (Ui.Res("AmberSoft"), Ui.Res("Amber")),
            "primary" => (Ui.Res("PrimarySoft"), Ui.Res("Primary")),
            _ => (Ui.Res("GreySoft"), Ui.Res("Text2")),
        };
        it.Line2 = string.Join(" · ", new[] { r.IsRl ? "强化学习" : "预训练", r.SizeText, r.Amp, $"{r.Created:MM-dd HH:mm}" }.Where(x => x.Length > 0));
        it.Progress = r.Steps > 0 ? Math.Min(1, r.CurrentStep / (double)r.Steps) : 0;
        var t = r.VsTeacher;
        var ev = r.Eval.LastOrDefault(p => !double.IsNaN(p.VsRule));
        if (t != null)
            it.Line3 = $"对老师得分 {t.Score} · " + (r.IsRl
                ? $"对起点 {r.Matches.FirstOrDefault(m => m.Opponent.StartsWith("初始"))?.Score ?? "-"}"
                : $"对随机 {r.Matches.FirstOrDefault(m => m.Opponent.StartsWith("随机"))?.Score ?? "-"}");
        else
            it.Line3 = $"第 {r.CurrentStep:N0} / {r.Steps:N0} {r.Unit}" + (r.IsRl && ev != null ? $" · 对规则 AI {ev.VsRule:0.000}" : "");
    }

    void ShowDetail(RunInfo? r)
    {
        Detail.Visibility = r == null ? Visibility.Collapsed : Visibility.Visible;
        NoRun.Visibility = r == null ? Visibility.Visible : Visibility.Collapsed;
        if (r == null) return;
        var task = Hub.TrainingOf(r.Dir);
        var (text, kind) = StatusOf(r);
        RunName.Text = r.Name;
        Ui.Badge(RunBadge, RunBadgeText, text, kind);
        var data = r.Args is { } a && a.TryGetProperty("data", out var d) ? string.Join(", ", d.EnumerateArray().Select(x => x.GetString())) : "";
        RunMeta.Text = string.Join(" · ", r.IsRl
            ? new[]
            {
                "强化学习（PPO 自对弈）", "起点 " + r.ArgStr("init").Replace('\\', '/'), r.SizeText, r.ShapeText, r.Amp,
                $"每轮 {r.Batch:N0} 样本", $"lr {r.ArgDouble("lr"):0.#####}", $"{r.Steps:N0} 轮",
                r.ArgStr("rules") == "duel" ? "二人" : "四国暗棋", r.RemoteHost.Length > 0 ? $"在 {r.RemoteHost} 上训练" : "",
            }
            : new[]
            {
                r.SizeText, r.ShapeText, r.Amp, $"batch {r.Batch}", $"lr {r.ArgDouble("lr"):0.####}", $"{r.Steps:N0} 步",
                r.ArgStr("rules") == "duel" ? "二人" : "四国暗棋", data, r.RemoteHost.Length > 0 ? $"在 {r.RemoteHost} 上训练" : "",
            }.Where(x => x.Length > 0));
        StopBtn.Visibility = task != null ? Visibility.Visible : Visibility.Collapsed;
        ResumeBtn.Visibility = task == null && r.CanResume ? Visibility.Visible : Visibility.Collapsed;
        PlayBtn.Visibility = r.HasModel ? Visibility.Visible : Visibility.Collapsed;

        // KPI
        Kpis.Children.Clear();
        int step = r.CurrentStep, steps = Math.Max(1, r.Steps);
        Kpi("进度", $"{100.0 * step / steps:0}%", $"共 {steps:N0} {r.Unit}");
        Kpi("速度", r.Speed > 0 ? $"{r.Speed:N0}" : "-", r.IsRl ? $"样本/秒 · 每轮 {r.Batch:N0}" : "样本/秒");
        if ((task != null || r.RemoteRunning) && step < steps)
            Kpi("预计剩余", r.Speed > 0 ? Ui.Duration((steps - step) * (double)r.Batch / r.Speed) : "-", r.IsRl ? "另加定期评估局" : "另加对战评估");
        else
            Kpi("用时", Ui.Duration((r.LastWrite - r.Created).TotalSeconds), $"{r.Created:MM-dd HH:mm} 开始");
        if (r.IsRl)
        {
            var evs = r.Eval.Where(p => !double.IsNaN(p.VsRule)).ToList();
            var first = evs.FirstOrDefault();
            var last = evs.LastOrDefault();
            Kpi("对规则 AI 得分", last != null ? last.VsRule.ToString("0.000") : "-",
                last != null ? $"第 {last.Step:N0} 轮 · 起点 {first!.VsRule:0.000}" : "尚未评估");
            Kpi("对起点模型得分", last != null ? last.VsBc.ToString("0.000") : "-",
                last != null ? $"胜负和 {last.VsBcWld}" : "尚未评估");
        }
        else
        {
            var ev = r.Eval.Count > 0 ? r.Eval[^1] : null;
            Kpi("留出损失", ev != null ? ev.Loss.ToString("0.000") : "-", ev != null ? $"第 {ev.Step:N0} 步" : "尚未评估");
            var t = r.VsTeacher;
            Kpi("对老师得分", t?.Score ?? "-", t != null ? $"胜负和 {t.Wld}" : "训练后出结果");
        }

        // 曲线：数据有变化才重画
        var key = $"{r.Name}|{r.Train.Count}|{r.Eval.Count}";
        if (key != _plotKey)
        {
            _plotKey = key;
            if (r.IsRl) DrawRl(r);
            else
            {
                Draw(PlotLoss, "损失（越低越好）", r, p => p.Loss, Alignment.UpperRight);
                Draw(PlotPolicy, "走法命中率：首选着法 = 老师着法", r, p => p.PolicyTop1, Alignment.LowerRight);
                Draw(PlotBelief, "猜子命中率：敌方暗子身份", r, p => p.BeliefTop1, Alignment.LowerRight);
                Draw(PlotValue, "胜负预测准确率", r, p => p.ValueAcc, Alignment.LowerRight);
            }
        }

        ResultCard.Visibility = r.HasSummary ? Visibility.Visible : Visibility.Collapsed;
        MatchGrid.ItemsSource = r.Matches;
        BeliefGrid.ItemsSource = r.Belief;
        BeliefSection.Visibility = r.Belief.Count > 0 ? Visibility.Visible : Visibility.Collapsed;

        var info = new FileInfo(r.OutFile);
        var ckey = $"{r.Name}|{(info.Exists ? info.Length : 0)}";
        if (ckey != _consoleKey)
        {
            _consoleKey = ckey;
            var atEnd = Console.VerticalOffset + Console.ViewportHeight >= Console.ExtentHeight - 4;
            Console.Text = info.Exists ? Procs.Tail(r.OutFile, 400) : "（没有输出文件）";
            if (atEnd || ckey.EndsWith("|0")) Console.ScrollToEnd();
        }
    }

    void Kpi(string label, string value, string sub)
    {
        var sp = new StackPanel();
        sp.Children.Add(new TextBlock { Text = label, Style = (Style)FindResource("Hint"), FontSize = 13 });
        sp.Children.Add(new TextBlock { Text = value, FontSize = 24, FontWeight = FontWeights.Bold, Margin = new Thickness(0, 4, 0, 2) });
        sp.Children.Add(new TextBlock { Text = sub, Style = (Style)FindResource("Hint"), TextTrimming = TextTrimming.CharacterEllipsis });
        var b = new Border { Child = sp, Style = (Style)FindResource("CardBox"), Padding = new Thickness(18, 14, 18, 14), Margin = new Thickness(0, 0, 12, 0) };
        Kpis.Children.Add(b);
    }

    static void StylePlot(WpfPlot wp)
    {
        var plt = wp.Plot;
        plt.Font.Set("Microsoft YaHei UI");
        plt.FigureBackground.Color = ScottPlot.Colors.White;
        plt.DataBackground.Color = ScottPlot.Colors.White;
        plt.Axes.Color(ScottPlot.Color.FromHex("#9AA0AE"));
        plt.Grid.MajorLineColor = ScottPlot.Color.FromHex("#EEF0F5");
        foreach (var ax in new ScottPlot.IAxis[] { plt.Axes.Bottom, plt.Axes.Left })
        {
            ax.TickLabelStyle.FontSize = 13;
            ax.Label.FontSize = 13;
        }
        plt.Legend.FontSize = 13;
        wp.Refresh();
    }

    static List<(double X, double Y)> Pts(IEnumerable<LogPoint> src, Func<LogPoint, double> f) =>
        src.Where(p => !double.IsNaN(f(p))).Select(p => ((double)p.Step, f(p))).ToList();

    /// <summary>滑动平均（窗口 w），让每轮波动较大的曲线看得出趋势。</summary>
    static List<(double X, double Y)> Smooth(List<(double X, double Y)> pts, int w)
    {
        var o = new List<(double X, double Y)>();
        for (int i = 0; i < pts.Count; i++)
        {
            int a = Math.Max(0, i - w + 1);
            o.Add((pts[i].X, pts.Skip(a).Take(i - a + 1).Average(p => p.Y)));
        }
        return o;
    }

    static void Draw(WpfPlot wp, string title, RunInfo r, Func<LogPoint, double> f, Alignment legend) =>
        DrawSeries(wp, title, "步数", legend, null,
            new Series("训练", CTrain, 2, 0, Pts(r.Train, f)),
            new Series("留出（未参与训练）", CEval, 1.5f, 8, Pts(r.Eval, f)));

    void DrawRl(RunInfo r)
    {
        DrawSeries(PlotLoss, "对战得分（贪心走法，定期评估）", "轮次", Alignment.LowerRight, 0.5,
            new Series("对规则 AI（老师）", CTrain, 2, 7, Pts(r.Eval, p => p.VsRule)),
            new Series("对起点模型", CEval, 2, 7, Pts(r.Eval, p => p.VsBc)));
        var roll = Pts(r.Train, p => p.RollRule);
        DrawSeries(PlotPolicy, "采样对局中对规则 AI 的得分", "轮次", Alignment.LowerRight, 0.5,
            new Series("每轮", CLight, 1, 0, roll),
            new Series("近 10 轮平均", CTrain, 2.5f, 0, Smooth(roll, 10)));
        DrawSeries(PlotBelief, "策略熵（越低走法越果断）", "轮次", Alignment.UpperRight, null,
            new Series("熵", CTrain, 2, 0, Pts(r.Train, p => p.Entropy)));
        DrawSeries(PlotValue, "偏离起点模型的程度（KL）", "轮次", Alignment.UpperLeft, null,
            new Series("KL", CEval, 2, 0, Pts(r.Train, p => p.KlRef)));
    }

    static void DrawSeries(WpfPlot wp, string title, string xLabel, Alignment legend, double? refLine, params Series[] series)
    {
        var plt = wp.Plot;
        plt.Clear();
        plt.Title(title);
        plt.Axes.Title.Label.FontSize = 17;
        plt.Axes.Title.Label.Bold = true;
        plt.Axes.Title.Label.ForeColor = ScottPlot.Color.FromHex("#1F2330");
        plt.XLabel(xLabel);
        int n = 0;
        foreach (var sr in series.Where(x => x.Pts.Count > 0))
        {
            var s = plt.Add.Scatter(sr.Pts.Select(p => p.X).ToArray(), sr.Pts.Select(p => p.Y).ToArray());
            s.Color = sr.Color; s.LineWidth = sr.Width; s.MarkerSize = sr.Marker; s.LegendText = sr.Label;
            n++;
        }
        if (refLine is { } y && n > 0)
        {
            var h = plt.Add.HorizontalLine(y);
            h.Color = ScottPlot.Color.FromHex("#9AA0AE");
            h.LineWidth = 1;
            h.LinePattern = LinePattern.Dashed;
        }
        if (n > 0) plt.ShowLegend(legend);
        plt.Axes.AutoScale();
        wp.Refresh();
    }

    void RunList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (RunList.SelectedItem is RunItem it && it.Name != _selected)
        {
            _selected = it.Name;
            _plotKey = _consoleKey = "";
            Detail.ScrollToTop();
            if (_runs.TryGetValue(it.Name, out var r)) ShowDetail(r);
        }
    }

    RunInfo? Current => _selected != null && _runs.TryGetValue(_selected, out var r) ? r : null;

    public void NewRun(RunInfo? template = null)
    {
        var dlg = new NewRunDialog(template) { Owner = Window.GetWindow(this) };
        if (dlg.ShowDialog() == true && dlg.StartedName != null)
        {
            _pendingSelect = dlg.StartedName;
            _ = Hub.Refresh();
        }
    }

    public void ShowNewRunDialogForShot(string file, bool rl = false)
    {
        var dlg = new NewRunDialog(null) { Owner = Window.GetWindow(this) };
        if (rl) dlg.ModeRl.IsChecked = true;
        dlg.ContentRendered += async (_, _) =>
        {
            await Task.Delay(800);
            MainWindow.SaveVisual((FrameworkElement)dlg.Content, file);
            dlg.Close();
        };
        dlg.Show();
    }

    void NewRun_Click(object sender, RoutedEventArgs e) => NewRun();
    void Clone_Click(object sender, RoutedEventArgs e) => NewRun(Current);

    async void Stop_Click(object sender, RoutedEventArgs e)
    {
        var r = Current;
        if (r == null || Hub.TrainingOf(r.Dir) is not { } t) return;
        var msg = r.IsRl
            ? $"停止训练 {r.Name}？\n\n之后可以点“继续训练”，从最近一次存档（第 {r.LastEvalStep:N0} 轮）接着跑，存档之后的几轮会重跑。"
            : $"停止训练 {r.Name}？\n\n已保存的模型（第 {r.LastEvalStep:N0} 步）会保留，但这次训练不能接着跑，只能重新开始。";
        if (!Ui.Confirm(msg, "停止训练")) return;
        Procs.Kill(t.Pid);
        await Task.Delay(500);
        await Hub.Refresh();
    }

    async void Resume_Click(object sender, RoutedEventArgs e)
    {
        var r = Current;
        if (r == null || !r.CanResume) return;
        var argv = r.Argv;
        if (argv.Count == 0) { Ui.Info("这次训练的 args.json 里没有记录启动参数，无法继续。", "继续训练"); return; }
        if (Hub.Trainings.Any() && !Ui.Confirm("已经有训练在跑。两个训练同时跑会互相抢显卡，各自都会变慢。仍然继续？", "继续训练")) return;
        try { Procs.StartDetached("junqi.train_rl", argv.Append("--resume"), r.OutFile, append: true); }
        catch (Exception ex) { Ui.Info("启动失败：" + ex.Message, "继续训练"); return; }
        await Task.Delay(1500);
        await Hub.Refresh();
    }

    async void Play_Click(object sender, RoutedEventArgs e)
    {
        var r = Current;
        if (r == null) return;
        if (Hub.Play is { } p) { PlayService.Stop(p); await Task.Delay(800); }
        PlayService.Start(r.ModelPath);
        await Task.Delay(800);
        await Hub.Refresh();
        MainWindow.Instance.ShowPlay();
    }

    void Folder_Click(object sender, RoutedEventArgs e) { if (Current is { } r) Procs.OpenFolder(r.Dir); }

    void OpenLog_Click(object sender, RoutedEventArgs e)
    {
        if (Current is { } r && File.Exists(r.OutFile))
            try { System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("notepad.exe", $"\"{r.OutFile}\"")); } catch { }
    }
}
