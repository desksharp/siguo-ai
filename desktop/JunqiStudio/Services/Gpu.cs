using System.Diagnostics;
using System.Management;
using System.Text.Json;

namespace JunqiStudio.Services;

/// <summary>显卡信息：名称来自 WMI，占用率与显存来自 Windows 性能计数器（GPU Engine / GPU Adapter Memory）。</summary>
public static class Gpu
{
    public static string Name { get; private set; } = "";
    public static string TorchInfo { get; private set; } = "";
    public static bool? TorchOk { get; private set; }
    public static double TotalGB { get; private set; }

    static readonly Dictionary<string, PerformanceCounter> Engines = new();
    static readonly Dictionary<string, PerformanceCounter> Memory = new();

    public static void DetectName()
    {
        try
        {
            using var q = new ManagementObjectSearcher("SELECT Name FROM Win32_VideoController");
            var names = q.Get().Cast<ManagementObject>().Select(o => o["Name"] as string ?? "").ToList();
            Name = names.FirstOrDefault(n => n.Contains("RX ") || n.Contains("GeForce") || n.Contains("Radeon Pro") || n.Contains("Arc"))
                   ?? names.FirstOrDefault() ?? "";
        }
        catch { }
    }

    /// <summary>用训练环境的 python 问 PyTorch 能否用显卡（约 3–5 秒）。</summary>
    public static async Task DetectTorch()
    {
        var code = "import json,torch;ok=torch.cuda.is_available();p=torch.cuda.get_device_properties(0) if ok else None;" +
                   "print(json.dumps({'ok':ok,'torch':torch.__version__,'hip':torch.version.hip,'name':p.name if p else '','mem':p.total_memory if p else 0}))";
        var lines = new List<string>();
        try
        {
            await Procs.RunCaptured(Settings.Current.Python, new[] { "-c", code }, Settings.Current.ProjectDir, l => { lock (lines) lines.Add(l); });
            var js = lines.LastOrDefault(l => l.StartsWith("{"));
            if (js == null) { TorchOk = false; TorchInfo = "PyTorch 不可用"; return; }
            var e = JsonDocument.Parse(js).RootElement;
            TorchOk = e.GetProperty("ok").GetBoolean();
            var hip = e.GetProperty("hip").ToString();
            var ver = e.GetProperty("torch").GetString() ?? "";
            TorchInfo = $"PyTorch {ver.Split('+')[0]}" + (hip.Length > 0 && hip != "null" ? $" · ROCm {string.Join('.', hip.Split('.').Take(2))}" : "");
            TotalGB = e.GetProperty("mem").GetInt64() / (double)(1L << 30);
            if (TorchOk == true && e.GetProperty("name").GetString() is { Length: > 0 } n) Name = n;
        }
        catch { TorchOk = false; TorchInfo = "PyTorch 不可用"; }
    }

    /// <summary>返回 (占用率 %, 已用显存 GB)。率类计数器第一次取值为 0，之后正常。</summary>
    public static (double util, double usedGB) Sample()
    {
        double util = 0, used = 0;
        try
        {
            var cat = new PerformanceCounterCategory("GPU Engine");
            var names = cat.GetInstanceNames();
            var live = names.ToHashSet();
            foreach (var k in Engines.Keys.Where(k => !live.Contains(k)).ToList()) { Engines[k].Dispose(); Engines.Remove(k); }
            var perType = new Dictionary<string, double>();
            foreach (var n in names)
            {
                if (!Engines.TryGetValue(n, out var c))
                    Engines[n] = c = new PerformanceCounter("GPU Engine", "Utilization Percentage", n, true);
                var i = n.IndexOf("_luid_");
                var j = n.IndexOf("_engtype_");
                if (i < 0 || j < 0) continue;
                var key = n.Substring(i, n.IndexOf("_phys_") is var k2 and > 0 ? k2 - i : 0) + n[j..];
                perType[key] = perType.GetValueOrDefault(key) + c.NextValue();
            }
            util = perType.Count > 0 ? Math.Min(100, perType.Values.Max()) : 0;

            var mcat = new PerformanceCounterCategory("GPU Adapter Memory");
            foreach (var n in mcat.GetInstanceNames())
            {
                if (!Memory.TryGetValue(n, out var c))
                    Memory[n] = c = new PerformanceCounter("GPU Adapter Memory", "Dedicated Usage", n, true);
                used = Math.Max(used, c.NextValue() / (1L << 30));
            }
        }
        catch { }
        return (util, used);
    }
}
