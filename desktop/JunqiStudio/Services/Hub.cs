using System.Windows;

namespace JunqiStudio.Services;

/// <summary>每 2 秒在后台扫描一次进程与显卡，然后在界面线程上通知各页面刷新。</summary>
public static class Hub
{
    public static List<PyTask> Tasks { get; private set; } = new();
    public static double GpuUtil { get; private set; }
    public static double GpuUsedGB { get; private set; }
    public static event Action? Updated;

    static CancellationTokenSource? _cts;

    public static void Start()
    {
        _cts = new CancellationTokenSource();
        var ct = _cts.Token;
        Task.Run(async () =>
        {
            Gpu.DetectName();
            while (!ct.IsCancellationRequested)
            {
                var tasks = Procs.Scan();
                var (u, m) = Gpu.Sample();
                Application.Current?.Dispatcher.Invoke(() =>
                {
                    Tasks = tasks;
                    GpuUtil = u;
                    GpuUsedGB = m;
                    Updated?.Invoke();
                });
                try { await Task.Delay(2000, ct); } catch { }
            }
        }, ct);
    }

    /// <summary>立即重扫一次（启动/停止任务后调用，不等下一个 2 秒周期）。</summary>
    public static async Task Refresh()
    {
        var tasks = await Task.Run(Procs.Scan);
        Tasks = tasks;
        Updated?.Invoke();
    }

    public static void Stop() => _cts?.Cancel();

    public static PyTask? TrainingOf(string runDir)
    {
        var proj = Settings.Current.ProjectDir;
        var key = Procs.Norm(runDir, proj);
        return Tasks.FirstOrDefault(t => t.Kind == "train" && Procs.Norm(t.Arg, proj) == key);
    }

    public static IEnumerable<PyTask> Trainings => Tasks.Where(t => t.Kind == "train");
    public static PyTask? Play => Tasks.FirstOrDefault(t => t.Kind == "play");
    public static PyTask? Gen => Tasks.FirstOrDefault(t => t.Kind == "gen");
}
