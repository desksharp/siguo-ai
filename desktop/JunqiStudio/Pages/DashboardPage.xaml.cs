using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using JunqiStudio.Services;

namespace JunqiStudio.Pages;

public partial class DashboardPage : UserControl, IPage
{
    RunInfo? _run;

    public DashboardPage()
    {
        InitializeComponent();
        Hub.Updated += () => { if (IsVisible) Refresh(); };
        Loaded += (_, _) => Refresh();
    }

    public void OnShown() => Refresh();

    void Refresh()
    {
        var s = Settings.Current;
        var training = Hub.Trainings.FirstOrDefault();
        var all = RunInfo.LoadAll();
        var remote = all.Where(r => r.RemoteRunning).ToList();
        _run = training != null ? RunInfo.Load(Procs.Norm(training.Arg, s.ProjectDir)) : remote.FirstOrDefault() ?? all.FirstOrDefault();
        if (_run != null && training != null) _run = RunInfo.Load(_run.Dir) ?? _run;
        FillTraining(training);
        FillPlay();
        FillGpu();
        FillCheck();

        var parts = new List<string>();
        int nTrain = Hub.Trainings.Count() + remote.Count;
        parts.Add(nTrain > 0 ? $"{nTrain} 个训练进行中" + (remote.Count > 0 ? $"（{remote.Count} 个在远程）" : "") : "没有正在进行的训练");
        parts.Add(Hub.Play != null ? "对弈服务运行中" : "对弈服务未启动");
        if (Hub.Gen != null) parts.Add("正在产数据");
        SumText.Text = string.Join(" · ", parts);
        SumDot.Fill = Ui.Res(nTrain > 0 || Hub.Play != null ? "Green" : "Grey");
    }

    void FillTraining(PyTask? task)
    {
        var r = _run;
        TrStopBtn.Visibility = task != null ? Visibility.Visible : Visibility.Collapsed;
        TrCurveBtn.IsEnabled = r != null;
        TrBadge2.Visibility = Visibility.Collapsed;
        TrMetrics.Children.Clear();
        if (r == null)
        {
            Ui.Badge(TrBadge, TrBadgeText, "空闲", "grey");
            Ui.Dot(TrRing, TrDot, "grey");
            TrName.Text = "还没有训练记录";
            TrMeta.Text = "";
            TrProgress.Value = 0;
            TrProgressText.Text = "";
            TrSpeed.Text = "点右上角“新建训练”开始第一次训练。";
            return;
        }
        int step = r.CurrentStep, steps = Math.Max(1, r.Steps);
        TrName.Text = r.Name;
        TrMeta.Text = string.Join(" · ", new[]
        {
            r.IsRl ? "强化学习" : "", r.SizeText, r.ShapeText, r.Amp,
            r.Batch > 0 ? (r.IsRl ? $"每轮 {r.Batch:N0} 样本" : $"batch {r.Batch}") : "",
        }.Where(x => x.Length > 0));
        TrProgress.Value = Math.Min(1, step / (double)steps);
        TrProgressText.Text = $"第 {step:N0} / {steps:N0} {r.Unit} · {100.0 * step / steps:0}%";

        var where = task == null && r.RemoteHost.Length > 0 ? $" · 在 {r.RemoteHost} 上训练" : "";
        if (task != null || r.RemoteRunning)
        {
            Ui.Badge(TrBadge, TrBadgeText, task != null ? "运行中" : "远程运行中", "green");
            Ui.Dot(TrRing, TrDot, "green");
            if (step >= steps)
            {
                TrBadge2.Visibility = Visibility.Visible;
                Ui.Badge(TrBadge2, TrBadge2Text, "对战评估中", "amber");
                TrSpeed.Text = r.IsRl
                    ? "训练轮数已完成，正在对随机 AI / 规则 AI / 起点模型下最终评估局，通常需要十几分钟。"
                    : "训练步数已完成，正在对随机 AI / 规则 AI 下评估局并生成信念报告，通常需要几分钟。";
            }
            else
            {
                var left = r.Speed > 0 ? (steps - step) * (double)r.Batch / r.Speed : double.NaN;
                TrSpeed.Text = $"{r.Speed:N0} 样本/秒 · 预计还需 {Ui.Duration(left)}（{(r.IsRl ? "另加定期评估局" : "之后另有几分钟对战评估")}） · 开始于 {r.Created:MM-dd HH:mm}{where}";
            }
        }
        else if (r.HasSummary)
        {
            Ui.Badge(TrBadge, TrBadgeText, "已完成", "primary");
            Ui.Dot(TrRing, TrDot, "primary");
            var t = r.VsTeacher;
            TrSpeed.Text = $"完成于 {r.LastWrite:MM-dd HH:mm}{where}" + (t != null ? $" · 对老师（规则 AI T=0.1）得分 {t.Score}，胜/负/和 {t.Wld}" : "");
        }
        else if (r.RemoteHost.Length > 0)
        {
            Ui.Badge(TrBadge, TrBadgeText, "远程无响应", "amber");
            Ui.Dot(TrRing, TrDot, "amber");
            TrSpeed.Text = $"在 {r.RemoteHost} 上训练，超过 10 分钟没有收到同步（中转脚本停了或连不上服务器），显示的是最后同步到的第 {step:N0} {r.Unit}。";
        }
        else
        {
            Ui.Badge(TrBadge, TrBadgeText, step > 0 ? "已中断" : "未开始", "amber");
            Ui.Dot(TrRing, TrDot, "amber");
            TrSpeed.Text = step > 0
                ? $"停在第 {step:N0} {r.Unit}" + (r.HasModel ? $"，已保存的模型是第 {r.LastEvalStep:N0} {r.Unit}的，可以拿来对弈" : "") + "。"
                  + (r.CanResume ? "在训练页点“继续训练”可以接着跑。" : "")
                : "训练进程没有在运行。";
        }

        if (r.IsRl)
        {
            var evs = r.Eval.Where(p => !double.IsNaN(p.VsRule)).ToList();
            var last = evs.LastOrDefault();
            if (last != null)
            {
                Chip("对规则 AI", last.VsRule.ToString("0.000"), $"起点 {evs[0].VsRule:0.000}");
                Chip("对起点模型", last.VsBc.ToString("0.000"), $"第 {last.Step:N0} 轮");
            }
            if (r.Train.Count > 0)
            {
                var p = r.Train[^1];
                Chip("策略熵", p.Entropy.ToString("0.00"), $"起步 {r.Train[0].Entropy:0.00}");
                Chip("胜负预测", p.ValueAcc.ToString("0.000"), null);
            }
        }
        else if (r.Train.Count > 0)
        {
            var p = r.Train[^1];
            var ev = r.Eval.Count > 0 ? r.Eval[^1] : null;
            Chip("损失", p.Loss.ToString("0.000"), ev != null ? $"留出 {ev.Loss:0.000}" : null);
            Chip("走法命中", p.PolicyTop1.ToString("0.000"), ev != null ? $"留出 {ev.PolicyTop1:0.000}" : null);
            Chip("猜子命中", p.BeliefTop1.ToString("0.000"), ev != null ? $"留出 {ev.BeliefTop1:0.000}" : null);
            Chip("胜负预测", p.ValueAcc.ToString("0.000"), ev != null ? $"留出 {ev.ValueAcc:0.000}" : null);
        }
    }

    void Chip(string label, string value, string? sub)
    {
        var sp = new StackPanel();
        sp.Children.Add(new TextBlock { Text = label, Style = (Style)FindResource("Hint") });
        sp.Children.Add(new TextBlock { Text = value, FontSize = 18, FontWeight = FontWeights.Bold, Margin = new Thickness(0, 2, 0, 0) });
        if (sub != null) sp.Children.Add(new TextBlock { Text = sub, Style = (Style)FindResource("Hint") });
        TrMetrics.Children.Add(new Border
        {
            Child = sp, Background = Ui.Res("Bg"), CornerRadius = new CornerRadius(10),
            Padding = new Thickness(14, 8, 18, 8), Margin = new Thickness(0, 0, 10, 0), MinWidth = 110,
        });
    }

    void FillPlay()
    {
        var s = Settings.Current;
        var p = Hub.Play;
        PlUrl.Text = $"http://127.0.0.1:{s.PlayPort}";
        if (p != null)
        {
            Ui.Badge(PlBadge, PlBadgeText, "运行中", "green");
            Ui.Dot(PlRing, PlDot, "green");
            PlToggleText.Text = "停止";
            PlToggleIcon.Text = "";
            var model = Procs.Norm(p.Arg, s.ProjectDir);
            PlMeta.Text = File.Exists(model) ? $"神经网络模型：{Path.GetFileName(Path.GetDirectoryName(model))}" : "只有规则 AI（没有找到神经网络模型）";
        }
        else
        {
            Ui.Badge(PlBadge, PlBadgeText, "已停止", "grey");
            Ui.Dot(PlRing, PlDot, "grey");
            PlToggleText.Text = "启动";
            PlToggleIcon.Text = "";
            PlMeta.Text = "";
        }
    }

    void FillGpu()
    {
        GpName.Text = string.Join(" · ", new[] { Gpu.Name, Gpu.TorchInfo }.Where(x => x.Length > 0));
        if (Gpu.TorchOk == true) { Ui.Badge(GpBadge, GpBadgeText, "可用", "green"); Ui.Dot(GpRing, GpDot, "green"); }
        else if (Gpu.TorchOk == false) { Ui.Badge(GpBadge, GpBadgeText, "PyTorch 无法使用显卡", "red"); Ui.Dot(GpRing, GpDot, "red"); }
        else { Ui.Badge(GpBadge, GpBadgeText, "检测中", "grey"); Ui.Dot(GpRing, GpDot, "grey"); }
        GpUtil.Value = Hub.GpuUtil;
        GpUtilText.Text = $"{Hub.GpuUtil:0}%";
        var total = Gpu.TotalGB > 0 ? Gpu.TotalGB : 24;
        GpMem.Value = Math.Min(1, Hub.GpuUsedGB / total);
        GpMemText.Text = $"{Hub.GpuUsedGB:0.0} / {total:0} GB";
        GpBench.Text = Settings.Current.LastBench.Length > 0 ? "上次测速：" + Settings.Current.LastBench : "还没有测过速度。测速约 1 分钟，训练进行中测出来会偏低。";
    }

    void FillCheck()
    {
        var s = Settings.Current;
        if (s.LastCheckOk == true) { Ui.Badge(CkBadge, CkBadgeText, "通过", "green"); Ui.Dot(CkRing, CkDot, "green"); }
        else if (s.LastCheckOk == false) { Ui.Badge(CkBadge, CkBadgeText, "未通过", "red"); Ui.Dot(CkRing, CkDot, "red"); }
        else { Ui.Badge(CkBadge, CkBadgeText, "未运行", "grey"); Ui.Dot(CkRing, CkDot, "grey"); }
        CkLast.Text = s.LastCheckTime.Length > 0 ? $"上次运行：{s.LastCheckTime}" : "改过引擎或 Python 代码后运行一次。训练或对弈服务运行时引擎文件被占用，需要先停掉它们。";
    }

    async void Refresh_Click(object sender, RoutedEventArgs e) => await Hub.Refresh();

    void NewRun_Click(object sender, RoutedEventArgs e) => MainWindow.Instance.NewRun();

    void TrName_Click(object sender, MouseButtonEventArgs e) { if (_run != null) MainWindow.Instance.ShowTraining(_run.Name); }
    void TrCurve_Click(object sender, RoutedEventArgs e) { if (_run != null) MainWindow.Instance.ShowTraining(_run.Name); }

    async void TrStop_Click(object sender, RoutedEventArgs e)
    {
        var t = Hub.Trainings.FirstOrDefault();
        if (t == null || _run == null) return;
        if (!Ui.Confirm($"停止训练 {_run.Name}？\n\n已保存的模型（第 {_run.LastEvalStep:N0} 步）会保留，但这次训练不能接着跑，只能重新开始。", "停止训练")) return;
        Procs.Kill(t.Pid);
        await Task.Delay(500);
        await Hub.Refresh();
    }

    void PlOpen_Click(object sender, RoutedEventArgs e) => MainWindow.Instance.ShowPlay();

    async void PlToggle_Click(object sender, RoutedEventArgs e)
    {
        if (Hub.Play is { } p) PlayService.Stop(p);
        else PlayService.Start(PlayService.DefaultModel());
        await Task.Delay(800);
        await Hub.Refresh();
    }

    async void Bench_Click(object sender, RoutedEventArgs e)
    {
        var s = Settings.Current;
        var script = Path.Combine(s.BenchDir, "bench.py");
        if (!File.Exists(script)) { MessageBox.Show($"找不到 {script}", "测速"); return; }
        if (Hub.Trainings.Any() && !Ui.Confirm("训练正在占用显卡，测出来的速度会明显偏低。继续？", "测速")) return;
        var w = new OutputWindow("显卡测速", "用真实训练数据测 5 种模型配置的训练速度，并核对 GPU 与 CPU 结果一致");
        var code = await w.Run(s.Python, new[] { "bench.py", "cuda" }, s.BenchDir);
        if (code != 0) return;
        var parts = new List<string>();
        lock (w.Lines)
            foreach (var l in w.Lines.Where(l => l.StartsWith("{")))
            {
                try
                {
                    var j = JsonDocument.Parse(l).RootElement;
                    if (!j.TryGetProperty("samples_per_s", out var sp)) continue;
                    var amp = j.GetProperty("amp").GetString();
                    var model = j.GetProperty("params_M").GetDouble();
                    if (amp == "bfloat16") parts.Add($"{model:0.00}M {sp.GetInt32():N0}/秒");
                }
                catch { }
            }
        s.LastBench = $"{DateTime.Now:MM-dd HH:mm} · bf16 batch 1024：" + string.Join("，", parts);
        s.Save();
        Refresh();
    }

    async void Check_Click(object sender, RoutedEventArgs e)
    {
        if (Hub.Trainings.Any() || Hub.Play != null)
        {
            MessageBox.Show("训练或对弈服务正在使用引擎文件 junqi.dll，编译时无法替换它。\n请先停止训练和对弈服务，再运行自检。", "引擎自检",
                            MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }
        var s = Settings.Current;
        var w = new OutputWindow("引擎自检", "编译 junqi.dll → 引擎自测 → 规则回归测试 → Python 测试");
        var code = await w.Run("cmd.exe", new[] { "/d", "/c", Path.Combine("scripts", "check_win.bat") }, s.ProjectDir,
                               c => c == 0 ? "全部通过" : "有测试没通过，看上面的输出",
                               new Dictionary<string, string> { ["PYTHON"] = s.Python });
        s.LastCheckOk = code == 0;
        s.LastCheckTime = DateTime.Now.ToString("yyyy-MM-dd HH:mm");
        s.Save();
        Refresh();
    }
}
