using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using JunqiStudio.Services;

namespace JunqiStudio.Pages;

public partial class NewRunDialog : Window
{
    public string? StartedName { get; private set; }
    bool _ready;

    // 7900 XTX 上 bf16、batch 1024 的实测速度（样本/秒，含数据管线）
    static readonly Dictionary<string, double> Speed = new() { ["small"] = 7800, ["medium"] = 3870, ["large"] = 1350 };

    // 7900 XTX 上中模型强化学习的实测：每轮 32768 样本约 32 秒（采样 + 复盘 + 两遍 PPO 更新）；800 局评估约 1 分钟
    const double RlSamplesPerSec = 1030, RlEvalGamesPerSec = 13;

    public NewRunDialog(RunInfo? template)
    {
        InitializeComponent();
        NameBox.Text = $"bc_four_{DateTime.Now:MMdd_HHmm}";
        if (template is { IsRl: true }) PrefillRl(template);
        else if (template != null) Prefill(template);
        LoadData(template);
        LoadInits(template);
        _ready = true;
        Mode_Changed(this, new RoutedEventArgs());
    }

    bool Rl => ModeRl.IsChecked == true;
    string Prefix => (Rl ? "rl_" : "bc_") + (Rules == "duel" ? "duel_" : "four_");

    void PrefillRl(RunInfo t)
    {
        ModeRl.IsChecked = true;
        var rules = t.ArgStr("rules");
        foreach (ComboBoxItem it in RulesBox.Items) it.IsSelected = (string)it.Tag == rules;
        NameBox.Text = $"rl_{(rules == "duel" ? "duel" : "four")}_{DateTime.Now:MMdd_HHmm}";
        ItersBox.Text = t.Steps.ToString();
        foreach (ComboBoxItem it in SamplesBox.Items) it.IsSelected = (string)it.Content == t.Batch.ToString();
        RlLrBox.Text = t.ArgDouble("lr").ToString("0.####e0", CultureInfo.InvariantCulture);
        foreach (ComboBoxItem it in RlAmpBox.Items) it.IsSelected = (string)it.Tag == (t.ArgStr("amp") is "bf16" ? "bf16" : "none");
        RlEvalBox.Text = t.ArgInt("eval_every").ToString();
        RlGamesBox.Text = t.ArgInt("eval_games").ToString();
    }

    bool BoardMatches(RunInfo r) => BoardMatches(r.Args is { } a && a.TryGetProperty("board", out var b) ? b : null);

    /// <summary>棋盘规则（弧线、中路铁路、营区后角）与所选规则文件一致才能混用（模型作起点、数据作训练）。
    /// 没记录这些字段的老训练、老数据按当时的默认（营区后角算直线）处理。</summary>
    bool BoardMatches(JsonElement? have)
    {
        try
        {
            var f = Path.Combine(Settings.Current.ProjectDir, "configs", "rules", Rules + ".json");
            var want = JsonDocument.Parse(File.ReadAllText(f)).RootElement;
            int W(string k, int dflt) => want.TryGetProperty(k, out var v) ? Num(v, dflt) : dflt;
            int H(string k, int dflt) => have is { } h && h.TryGetProperty(k, out var v) ? Num(v, dflt) : dflt;
            return H("arc_mode", 2) == W("arc_mode", 2) && H("center_middle_rail", 1) == W("center_middle_rail", 1)
                && H("camp_corner_straight", 1) == W("camp_corner_straight", 0);
        }
        catch { return false; }
    }

    /// <summary>规则文件里的数值，或 arc_mode 这类用名字写的枚举。</summary>
    static int Num(JsonElement v, int dflt) => v.ValueKind switch
    {
        JsonValueKind.Number => v.TryGetInt32(out var i) ? i : dflt,
        JsonValueKind.String => Array.IndexOf(new[] { "none", "front_rows", "side_rails", "both" }, v.GetString()) is var j and >= 0 ? j : dflt,
        _ => dflt,
    };

    void LoadInits(RunInfo? template = null)
    {
        InitBox.Items.Clear();
        var runs = RunInfo.LoadAll().Where(r => r.HasModel && r.ArgStr("rules") == Rules && BoardMatches(r)).ToList();
        var want = template?.ArgStr("init").Replace('\\', '/');
        ComboBoxItem? pick = null;
        double best = -1;
        foreach (var r in runs)
        {
            var t = r.VsTeacher;
            var ev = r.Eval.LastOrDefault(p => !double.IsNaN(p.VsRule));
            var score = t != null ? t.Score : ev != null ? ev.VsRule.ToString("0.000") : "-";
            var item = new ComboBoxItem
            {
                Content = $"{r.Name}    {(r.IsRl ? "强化学习" : "预训练")} · {r.SizeText} · 对老师 {score}{(r.HasSummary ? "" : "（未完成）")}",
                Tag = r,
            };
            InitBox.Items.Add(item);
            var rel = $"runs/{r.Name}/model.pt";
            if (want != null && want == rel) { pick = item; best = 2; }
            if (best < 2 && r.HasSummary && double.TryParse(score, NumberStyles.Float, CultureInfo.InvariantCulture, out var sc) && sc > best)
            {
                pick = item;
                best = sc;
            }
        }
        pick ??= InitBox.Items.Count > 0 ? (ComboBoxItem)InitBox.Items[0] : null;
        if (pick != null) pick.IsSelected = true;
        NoInit.Visibility = runs.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    RunInfo? Init => (InitBox.SelectedItem as ComboBoxItem)?.Tag as RunInfo;

    void Mode_Changed(object sender, RoutedEventArgs e)
    {
        if (!_ready) return;
        BcPanel.Visibility = Rl ? Visibility.Collapsed : Visibility.Visible;
        RlPanel.Visibility = Rl ? Visibility.Visible : Visibility.Collapsed;
        SubText.Text = Rl
            ? "强化学习：从一个已训练好的模型出发，让它和自己、历史存档、规则 AI 对下，按输赢改进，目标是超过规则 AI。训练在后台进行，关掉工作台也不会停。"
            : "行为克隆预训练：模仿规则 AI 的走法，同时学习猜敌方暗子身份和预测胜负。训练在后台进行，关掉工作台也不会停。";
        if (Regex.IsMatch(NameBox.Text, @"^(bc|rl)_(four|duel)_\d{4}_\d{4}$")) NameBox.Text = Prefix + DateTime.Now.ToString("MMdd_HHmm");
        Update();
    }

    void Prefill(RunInfo t)
    {
        var rules = t.ArgStr("rules");
        foreach (ComboBoxItem it in RulesBox.Items) it.IsSelected = (string)it.Tag == rules;
        if (rules == "duel") NameBox.Text = $"bc_duel_{DateTime.Now:MMdd_HHmm}";
        int d = t.ArgInt("d_model"), l = t.ArgInt("layers"), h = t.ArgInt("heads");
        var preset = (d, l, h) switch
        {
            (128, 4, 4) => PSmall,
            (192, 6, 6) => PMedium,
            (256, 8, 8) => PLarge,
            _ => PCustom,
        };
        preset.IsChecked = true;
        DBox.Text = d.ToString(); LBox.Text = l.ToString(); HBox.Text = h.ToString();
        StepsBox.Text = t.Steps.ToString();
        foreach (ComboBoxItem it in BatchBox.Items) it.IsSelected = (string)it.Content == t.Batch.ToString();
        LrBox.Text = t.ArgDouble("lr").ToString("0.####e0", CultureInfo.InvariantCulture);
        foreach (ComboBoxItem it in AmpBox.Items) it.IsSelected = (string)it.Tag == (t.ArgStr("amp") is "bf16" ? "bf16" : "none");
        EvalBox.Text = t.ArgInt("eval_every").ToString();
        GamesBox.Text = t.ArgInt("match_games").ToString();
    }

    string Rules => (RulesBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "siguo_standard";

    void LoadData(RunInfo? template = null)
    {
        var chosen = template?.Args is { } a && a.TryGetProperty("data", out var d)
            ? d.EnumerateArray().Select(x => (x.GetString() ?? "").Replace('\\', '/')).ToHashSet()
            : null;
        DataList.Children.Clear();
        var sets = DataSet.LoadAll().Where(s => s.Rules == Rules).ToList();
        foreach (var s in sets)
        {
            var ok = BoardMatches(s.Config);
            var cb = new CheckBox
            {
                Content = new TextBlock
                {
                    Text = $"{s.RelPath}    {s.GamesText} 局 · {s.PliesText} 步 · {s.Shards} 个分片 · {s.SizeText}" + (ok ? "" : " · 旧规则"),
                    Foreground = ok ? Ui.Res("Text") : Ui.Res("Text2"),
                },
                Tag = s, Margin = new Thickness(0, 4, 0, 4),
                IsChecked = chosen?.Contains(s.RelPath) ?? ok,
            };
            cb.Checked += Any_Changed;
            cb.Unchecked += Any_Changed;
            DataList.Children.Add(cb);
        }
        NoData.Visibility = sets.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    List<DataSet> ChosenData => DataList.Children.OfType<CheckBox>().Where(c => c.IsChecked == true).Select(c => (DataSet)c.Tag).ToList();

    (int d, int l, int h, string preset) Shape()
    {
        if (PSmall.IsChecked == true) return (128, 4, 4, "small");
        if (PLarge.IsChecked == true) return (256, 8, 8, "large");
        if (PCustom.IsChecked == true)
            return (int.TryParse(DBox.Text, out var d) ? d : 0, int.TryParse(LBox.Text, out var l) ? l : 0, int.TryParse(HBox.Text, out var h) ? h : 0, "custom");
        return (192, 6, 6, "medium");
    }

    int Batch => int.TryParse((BatchBox.SelectedItem as ComboBoxItem)?.Content as string, out var b) ? b : 1024;
    string Amp => (AmpBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "bf16";

    int Samples => int.TryParse((SamplesBox.SelectedItem as ComboBoxItem)?.Content as string, out var b) ? b : 32768;

    List<string> RlArgs(string name)
    {
        var init = Init != null ? $"runs/{Init.Name}/model.pt" : "?";
        return new List<string>
        {
            "--init", init, "--out", $"runs/{name}", "--rules", Rules, "--iters", ItersBox.Text.Trim(),
            "--samples", Samples.ToString(), "--lr", RlLrBox.Text.Trim(),
            "--amp", (RlAmpBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "bf16",
            "--eval-every", RlEvalBox.Text.Trim(), "--eval-games", RlGamesBox.Text.Trim(),
        };
    }

    List<string> Args(string name)
    {
        var (d, l, h, _) = Shape();
        var a = new List<string> { "--data" };
        a.AddRange(ChosenData.Select(s => s.RelPath));
        a.AddRange(new[]
        {
            "--out", $"runs/{name}", "--rules", Rules, "--steps", StepsBox.Text.Trim(), "--eval-every", EvalBox.Text.Trim(),
            "--d-model", d.ToString(), "--layers", l.ToString(), "--heads", h.ToString(), "--batch", Batch.ToString(),
            "--amp", Amp, "--lr", LrBox.Text.Trim(), "--warmup", "500", "--match-games", GamesBox.Text.Trim(),
        });
        return a;
    }

    void Rules_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (!_ready) return;
        if (Regex.IsMatch(NameBox.Text, @"^(bc|rl)_(four|duel)_\d{4}_\d{4}$")) NameBox.Text = Prefix + DateTime.Now.ToString("MMdd_HHmm");
        LoadData();
        LoadInits();
        Update();
    }

    void Any_Changed(object sender, RoutedEventArgs e) { if (_ready) Update(); }

    string? Validate()
    {
        var name = NameBox.Text.Trim();
        if (!Regex.IsMatch(name, @"^[A-Za-z0-9_\-\.]+$")) return "名称只能用英文字母、数字、下划线、横线和点。";
        if (Directory.Exists(Path.Combine(Settings.Current.RunsDir, name))) return $"runs\\{name} 已经存在，换个名称（避免覆盖之前的模型）。";
        if (Rl)
        {
            if (Init == null) return "选一个起点模型。";
            if (!int.TryParse(ItersBox.Text, out var it) || it < 1) return "轮数至少 1。";
            if (!double.TryParse(RlLrBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var rlr) || rlr <= 0 || rlr > 1e-3)
                return "学习率格式不对，强化学习一般用 1e-5 到 5e-5。";
            if (!int.TryParse(RlEvalBox.Text, out var re) || re < 1) return "评估间隔至少 1 轮。";
            if (!int.TryParse(RlGamesBox.Text, out var rg) || rg < 2) return "评估局数至少 2。";
            return null;
        }
        var (d, l, h, _) = Shape();
        if (d <= 0 || l <= 0 || h <= 0 || d % h != 0) return "模型维度必须是正整数，且 d-model 能被 heads 整除。";
        if (ChosenData.Count == 0) return "至少选一个训练数据目录。";
        if (ChosenData.Sum(s => s.Shards) < 2) return "所选数据至少要有 2 个分片（最后一个留作评估）。";
        if (!int.TryParse(StepsBox.Text, out var st) || st < 100) return "训练步数至少 100。";
        if (!double.TryParse(LrBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var lr) || lr <= 0 || lr > 0.01) return "学习率格式不对，例如 6e-4 或 0.0003。";
        if (!int.TryParse(EvalBox.Text, out var ev) || ev < 100) return "评估间隔至少 100 步。";
        if (!int.TryParse(GamesBox.Text, out var g) || g < 2) return "对战局数至少 2。";
        return null;
    }

    void Update()
    {
        CustomPanel.Visibility = PCustom.IsChecked == true ? Visibility.Visible : Visibility.Collapsed;
        var err = Validate();
        ErrorText.Text = err ?? "";
        var nm = NameBox.Text.Trim();
        if (Rl)
        {
            int.TryParse(ItersBox.Text, out var iters);
            int.TryParse(RlEvalBox.Text, out var every);
            int.TryParse(RlGamesBox.Text, out var games);
            // 按起点模型大小相对中模型的预训练速度折算
            var scale = Init is { } ir ? (ir.ArgInt("d_model"), ir.ArgInt("layers")) switch
            {
                (128, 4) => Speed["small"] / Speed["medium"],
                (256, 8) => Speed["large"] / Speed["medium"],
                _ => 1.0,
            } : 1.0;
            var amp = (RlAmpBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "bf16";
            var sps = RlSamplesPerSec * scale * (amp == "bf16" ? 1 : 0.6);
            var train = (double)iters * Samples / sps;
            var evals = every > 0 ? iters / every + 1 : 1;
            var evalSec = evals * 2.0 * games / RlEvalGamesPerSec + 3500 / RlEvalGamesPerSec;
            EstimateText.Text = $"共 {iters * (long)Samples / 1e6:0.0} 百万个样本、{evals} 次评估，预计 {Ui.Duration(train + evalSec)}";
            EstimateSub.Text = $"按 7900 XTX 实测估算：训练约 {Ui.Duration(train)}，评估约 {Ui.Duration(evalSec)}（含结束时 1000 局的最终评估）。训练时显卡会满载。";
            CmdText.Text = "python -m junqi.train_rl " + string.Join(" ", RlArgs(nm.Length > 0 ? nm : "?"));
            return;
        }
        var (_, _, _, preset) = Shape();
        int.TryParse(StepsBox.Text, out var steps);
        long samples = (long)steps * Batch;
        if (Speed.TryGetValue(preset, out var sp))
        {
            if (Amp != "bf16") sp *= 0.6;
            if (Batch < 1024) sp *= 0.9;
            var sec = samples / sp;
            EstimateText.Text = $"约 {samples / 1e6:0.0} 百万个样本，预计 {Ui.Duration(sec)}";
            EstimateSub.Text = $"按 7900 XTX 实测约 {sp:N0} 样本/秒估算；结束后对战评估另需几分钟。训练时显卡会满载。";
        }
        else
        {
            EstimateText.Text = $"约 {samples / 1e6:0.0} 百万个样本";
            EstimateSub.Text = "自定义模型无法估算用时，开始后看实际速度。";
        }
        var name = NameBox.Text.Trim();
        CmdText.Text = "python -m junqi.train_bc " + string.Join(" ", Args(name.Length > 0 ? name : "?"));
    }

    async void Start_Click(object sender, RoutedEventArgs e)
    {
        var err = Validate();
        if (err != null) { ErrorText.Text = err; return; }
        if (Hub.Trainings.Any() && !Ui.Confirm("已经有训练在跑。两个训练同时跑会互相抢显卡，各自都会变慢。仍然开始？", "新建训练")) return;
        var name = NameBox.Text.Trim();
        var dir = Path.Combine(Settings.Current.RunsDir, name);
        try
        {
            if (Rl) Procs.StartDetached("junqi.train_rl", RlArgs(name), Path.Combine(dir, "train.out"));
            else Procs.StartDetached("junqi.train_bc", Args(name), Path.Combine(dir, "train.out"));
        }
        catch (Exception ex)
        {
            ErrorText.Text = "启动失败：" + ex.Message;
            return;
        }
        StartedName = name;
        await Task.Delay(300);
        DialogResult = true;
    }
}
