using System.IO;
using System.Text.Json;

namespace JunqiStudio.Services;

public sealed class LogPoint
{
    public int Step;
    public double Loss, PolicyTop1, BeliefTop1, ValueAcc, SamplesPerSec, Lr;
    // 强化学习：每轮的采样与更新统计、评估局得分
    public double Entropy, KlRef, RollRule, Plies, VsRule, VsBc;
    public string VsRuleWld = "", VsBcWld = "";
}

public sealed class Match
{
    public string Opponent { get; init; } = "";
    public int Games { get; init; }
    public string Wld { get; init; } = "";
    public string Score { get; init; } = "";
    public string Elo { get; init; } = "";
}

public sealed class BeliefRow
{
    public string Progress { get; init; } = "";
    public string BaseTop1 { get; init; } = "";
    public string NeuralTop1 { get; init; } = "";
    public string BaseNll { get; init; } = "";
    public string NeuralNll { get; init; } = "";
    public string Marshal { get; init; } = "";
    public string Flag { get; init; } = "";
}

/// <summary>一次训练（runs/xxx 目录）：args.json + log.jsonl + summary.json + train.out。</summary>
public sealed class RunInfo
{
    public string Name { get; init; } = "";
    public string Dir { get; init; } = "";
    public DateTime Created { get; init; }
    public JsonElement? Args;
    public List<LogPoint> Train { get; } = new();
    public List<LogPoint> Eval { get; } = new();
    public List<Match> Matches { get; } = new();
    public List<BeliefRow> Belief { get; } = new();
    public bool HasSummary;
    public bool HasModel;
    public DateTime LastWrite;
    /// <summary>在别的机器上训练（如租用的 4090），日志由中转脚本同步过来，并每几分钟写一次 remote.json。</summary>
    public string RemoteHost = "";
    /// <summary>remote.json 说在跑，且 10 分钟内更新过（中转脚本停了就不再算在跑）。</summary>
    public bool RemoteRunning;

    /// <summary>强化学习（junqi.train_rl）：步数 = 轮数，batch = 每轮样本数。</summary>
    public bool IsRl => ArgStr("algo") == "ppo";
    public string Unit => IsRl ? "轮" : "步";
    public int Steps => ArgInt("steps");
    public int Batch => IsRl ? ArgInt("samples") : ArgInt("batch");
    public string CkptPath => Path.Combine(Dir, "ckpt.pt");
    public bool CanResume => IsRl && !HasSummary && RemoteHost.Length == 0 && File.Exists(CkptPath);
    public List<string> Argv => Args is { } a && a.TryGetProperty("argv", out var v) && v.ValueKind == JsonValueKind.Array
        ? v.EnumerateArray().Select(x => x.GetString() ?? "").ToList() : new();
    public long Params => Args is { } a && a.TryGetProperty("params", out var p) && p.TryGetInt64(out var v) ? v : 0;
    public int CurrentStep => Train.Count > 0 ? Train[^1].Step : 0;
    public int LastEvalStep => Eval.Count > 0 ? Eval[^1].Step : 0;
    public double Speed => Train.Count > 0 ? Train[^1].SamplesPerSec : 0;
    public string Amp => ArgStr("amp") is { Length: > 0 } s && s != "none" ? s : "fp32";
    public string ModelPath => Path.Combine(Dir, "model.pt");
    public string OutFile => Path.Combine(Dir, "train.out");

    public int ArgInt(string k) => Args is { } a && a.TryGetProperty(k, out var p) && p.TryGetInt32(out var v) ? v : 0;
    public double ArgDouble(string k) => Args is { } a && a.TryGetProperty(k, out var p) && p.TryGetDouble(out var v) ? v : 0;
    public string ArgStr(string k) => Args is { } a && a.TryGetProperty(k, out var p) ? p.ToString() : "";

    public string SizeText => Params > 0 ? $"{Params / 1e6:0.00}M 参数" : "";
    public string ShapeText => ArgInt("d_model") > 0 ? $"d{ArgInt("d_model")} · L{ArgInt("layers")} · H{ArgInt("heads")}" : "";

    public Match? VsTeacher => Matches.FirstOrDefault(m => m.Opponent.Contains("T=0.1"));

    public static RunInfo? Load(string dir)
    {
        if (!Directory.Exists(dir)) return null;
        var argsFile = Path.Combine(dir, "args.json");
        if (!File.Exists(argsFile)) return null;
        var r = new RunInfo { Name = Path.GetFileName(dir), Dir = dir, Created = File.GetLastWriteTime(argsFile) };
        try { r.Args = JsonDocument.Parse(ReadShared(argsFile)).RootElement.Clone(); } catch { }
        r.LastWrite = File.GetLastWriteTime(argsFile);
        var log = Path.Combine(dir, "log.jsonl");
        if (File.Exists(log))
        {
            r.LastWrite = File.GetLastWriteTime(log);
            foreach (var line in ReadShared(log).Split('\n'))
            {
                if (line.Length < 2) continue;
                try
                {
                    var e = JsonDocument.Parse(line).RootElement;
                    int step = e.GetProperty("step").GetInt32();
                    if (e.TryGetProperty("eval", out var ev)) r.Eval.Add(Point(step, ev));
                    else r.Train.Add(Point(step, e));
                }
                catch { }
            }
        }
        r.HasModel = File.Exists(r.ModelPath);
        var remote = Path.Combine(dir, "remote.json");
        if (File.Exists(remote))
        {
            try
            {
                var e = JsonDocument.Parse(ReadShared(remote)).RootElement;
                r.RemoteHost = e.TryGetProperty("host", out var h) ? h.GetString() ?? "远程" : "远程";
                r.RemoteRunning = e.TryGetProperty("running", out var run) && run.ValueKind == JsonValueKind.True
                                  && (DateTime.Now - File.GetLastWriteTime(remote)).TotalMinutes < 10;
            }
            catch { }
        }
        var sum = Path.Combine(dir, "summary.json");
        if (File.Exists(sum))
        {
            try
            {
                var s = JsonDocument.Parse(ReadShared(sum)).RootElement;
                r.HasSummary = true;
                foreach (var p in s.EnumerateObject())
                {
                    if (!p.Name.StartsWith("vs_")) continue;
                    var v = p.Value;
                    int w = v.GetProperty("w").GetInt32(), l = v.GetProperty("l").GetInt32(), d = v.GetProperty("d").GetInt32();
                    double elo = v.GetProperty("elo").GetDouble();
                    r.Matches.Add(new Match
                    {
                        Opponent = OpponentName(p.Name[3..]),
                        Games = w + l + d,
                        Wld = $"{w} / {l} / {d}",
                        Score = v.GetProperty("score").GetDouble().ToString("0.000"),
                        Elo = (elo >= 0 ? "+" : "") + elo.ToString("0"),
                    });
                }
                if (s.TryGetProperty("belief_baseline", out var bb) && s.TryGetProperty("belief_neural", out var bn))
                {
                    foreach (var p in bb.EnumerateObject())
                    {
                        if (!bn.TryGetProperty(p.Name, out var n)) continue;
                        var b = p.Value;
                        r.Belief.Add(new BeliefRow
                        {
                            Progress = double.TryParse(p.Name, out var f) ? $"{f * 100:0}%" : p.Name,
                            BaseTop1 = D(b, "top1"), NeuralTop1 = D(n, "top1"),
                            BaseNll = D(b, "nll"), NeuralNll = D(n, "nll"),
                            Marshal = $"{D(b, "p(司令)")} → {D(n, "p(司令)")}",
                            Flag = $"{D(b, "p(军旗)")} → {D(n, "p(军旗)")}",
                        });
                    }
                }
            }
            catch { }
        }
        return r;
    }

    static string D(JsonElement e, string k) => e.TryGetProperty(k, out var v) && v.TryGetDouble(out var d) ? d.ToString("0.000") : "-";

    static string OpponentName(string k) => k switch
    {
        "random" => "随机 AI",
        "rule(T=0.1)" => "规则 AI（T=0.1，老师）",
        "rule(T=0)" => "规则 AI（T=0，贪心）",
        "bc" => "初始模型（强化学习的起点）",
        _ => k
    };

    static LogPoint Point(int step, JsonElement e) => new()
    {
        Step = step,
        Loss = G(e, "loss"),
        PolicyTop1 = G(e, "policy_top1"),
        BeliefTop1 = G(e, "belief_top1"),
        ValueAcc = G(e, "value_acc"),
        SamplesPerSec = G(e, "samples_per_s"),
        Lr = G(e, "lr"),
        Entropy = G(e, "entropy"),
        KlRef = G(e, "kl_ref"),
        RollRule = G(e, "roll_rule"),
        Plies = G(e, "plies"),
        VsRule = G(e, "vs_rule"),
        VsBc = G(e, "vs_bc"),
        VsRuleWld = Wld(e, "vs_rule_wld"),
        VsBcWld = Wld(e, "vs_bc_wld"),
    };

    static string Wld(JsonElement e, string k) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Array
        ? string.Join(" / ", v.EnumerateArray().Select(x => x.ToString())) : "";

    static double G(JsonElement e, string k) => e.TryGetProperty(k, out var v) && v.TryGetDouble(out var d) ? d : double.NaN;

    public static string ReadShared(string file)
    {
        using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        using var sr = new StreamReader(fs);
        return sr.ReadToEnd();
    }

    public static List<RunInfo> LoadAll()
    {
        var root = Settings.Current.RunsDir;
        if (!Directory.Exists(root)) return new();
        return Directory.GetDirectories(root).Select(Load).Where(r => r != null).Cast<RunInfo>()
            .OrderByDescending(r => r.Created).ToList();
    }
}

/// <summary>自对弈数据目录（含 manifest.json）。</summary>
public sealed class DataSet
{
    public string Name { get; init; } = "";
    public string RelPath { get; init; } = "";
    public string Rules { get; init; } = "";
    public long Games { get; init; }
    public long Plies { get; init; }
    public int Shards { get; init; }
    public long Bytes { get; init; }
    public string Created { get; init; } = "";
    /// <summary>manifest 里记录的规则配置（含 camp_corner_straight 等棋盘规则）。</summary>
    public JsonElement? Config { get; init; }
    public string GamesText => Games.ToString("N0");
    public string PliesText => Plies >= 1_000_000 ? $"{Plies / 1e6:0.0}M" : Plies.ToString("N0");
    public string SizeText => Bytes >= 1 << 30 ? $"{Bytes / (double)(1 << 30):0.0} GB" : $"{Bytes / (double)(1 << 20):0} MB";
    public string RulesText => Rules == "siguo_standard" ? "四国暗棋" : Rules == "duel" ? "二人" : Rules;

    public static List<DataSet> LoadAll()
    {
        var root = Settings.Current.DataDir;
        var proj = Settings.Current.ProjectDir;
        var list = new List<DataSet>();
        if (!Directory.Exists(root)) return list;
        foreach (var mf in Directory.EnumerateFiles(root, "manifest.json", SearchOption.AllDirectories))
        {
            try
            {
                var dir = Path.GetDirectoryName(mf)!;
                var e = JsonDocument.Parse(RunInfo.ReadShared(mf)).RootElement;
                long games = 0, plies = 0;
                int shards = 0;
                if (e.TryGetProperty("shards", out var sh))
                    foreach (var s in sh.EnumerateArray())
                    {
                        shards++;
                        foreach (var k in new[] { "team0", "team1", "draws" })
                            if (s.TryGetProperty(k, out var v)) games += v.GetInt64();
                        if (s.TryGetProperty("plies", out var p)) plies += p.GetInt64();
                    }
                var files = Directory.GetFiles(dir, "*.jqr");
                list.Add(new DataSet
                {
                    Name = Path.GetRelativePath(root, dir).Replace('\\', '/'),
                    RelPath = Path.GetRelativePath(proj, dir).Replace('\\', '/'),
                    Rules = e.TryGetProperty("rules", out var r) ? r.GetString() ?? "" : "",
                    Games = games,
                    Plies = plies,
                    Shards = shards > 0 ? shards : files.Length,
                    Bytes = files.Sum(f => new FileInfo(f).Length),
                    Created = File.GetLastWriteTime(mf).ToString("yyyy-MM-dd HH:mm"),
                    Config = e.TryGetProperty("config", out var c) ? c.Clone() : null,
                });
            }
            catch { }
        }
        return list.OrderBy(d => d.Name).ToList();
    }
}
