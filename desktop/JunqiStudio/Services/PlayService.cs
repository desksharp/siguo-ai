using System.IO;

namespace JunqiStudio.Services;

/// <summary>人机对弈服务（python -m junqi.play）的启动与停止。</summary>
public static class PlayService
{
    /// <summary>上次选的模型；没有就用最新一个有 model.pt 的训练。</summary>
    public static string DefaultModel()
    {
        var s = Settings.Current;
        if (File.Exists(s.PlayModel)) return s.PlayModel;
        return RunInfo.LoadAll().FirstOrDefault(r => r.HasModel)?.ModelPath ?? "";
    }

    public static void Start(string model)
    {
        var s = Settings.Current;
        var args = new List<string> { "--port", s.PlayPort.ToString() };
        if (model.Length > 0) { args.Add("--model"); args.Add(model); }
        Procs.StartDetached("junqi.play", args, Path.Combine(s.LogsDir, "play.out"));
        s.PlayModel = model;
        s.Save();
    }

    public static void Stop(PyTask p) => Procs.Kill(p.Pid);

    public static string Url => $"http://127.0.0.1:{Settings.Current.PlayPort}";
}
