using System.Diagnostics;
using System.IO;
using System.Management;
using System.Text;
using System.Text.RegularExpressions;

namespace JunqiStudio.Services;

/// <summary>一个正在运行的 junqi Python 任务（按命令行识别，不依赖本程序是否启动过它）。</summary>
public sealed record PyTask(int Pid, string Kind, string CommandLine, string Arg)
{
    /// <summary>训练任务的 --out、对弈服务的 --model（原样）。</summary>
    public string Arg { get; } = Arg;
}

public static class Procs
{
    /// <summary>扫描所有 python.exe，找出 junqi.train_bc / junqi.train_rl / junqi.play / junqi.gen_data。
    /// venv 的 python.exe 会再拉起一个真正的解释器，两者命令行相同，按 (类型, 参数) 去重，保留父进程。</summary>
    public static List<PyTask> Scan()
    {
        var found = new List<(int pid, int ppid, string cmd)>();
        try
        {
            using var q = new ManagementObjectSearcher("SELECT ProcessId, ParentProcessId, CommandLine FROM Win32_Process WHERE Name='python.exe'");
            foreach (ManagementObject o in q.Get())
            {
                var cmd = o["CommandLine"] as string;
                if (string.IsNullOrEmpty(cmd) || !cmd.Contains("junqi.")) continue;
                found.Add((Convert.ToInt32(o["ProcessId"]), Convert.ToInt32(o["ParentProcessId"]), cmd));
            }
        }
        catch { }
        var pids = found.Select(f => f.pid).ToHashSet();
        var tasks = new List<PyTask>();
        foreach (var (pid, ppid, cmd) in found)
        {
            if (pids.Contains(ppid)) continue;  // 子解释器
            string kind = cmd.Contains("junqi.train_bc") || cmd.Contains("junqi.train_rl") ? "train" : cmd.Contains("junqi.play") ? "play" : cmd.Contains("junqi.gen_data") ? "gen" : "other";
            string arg = kind switch
            {
                "train" => ArgValue(cmd, "--out"),
                "play" => ArgValue(cmd, "--model"),
                "gen" => ArgValue(cmd, "--out"),
                _ => ""
            };
            tasks.Add(new PyTask(pid, kind, cmd, arg));
        }
        return tasks;
    }

    public static string ArgValue(string cmd, string name)
    {
        var m = Regex.Match(cmd, Regex.Escape(name) + "\\s+(\"([^\"]*)\"|(\\S+))");
        return m.Success ? (m.Groups[2].Success ? m.Groups[2].Value : m.Groups[3].Value) : "";
    }

    /// <summary>把 runs/xxx、runs\xxx、绝对路径统一成绝对路径，便于比较。</summary>
    public static string Norm(string path, string baseDir)
    {
        if (string.IsNullOrWhiteSpace(path)) return "";
        try { return Path.GetFullPath(Path.IsPathRooted(path) ? path : Path.Combine(baseDir, path)).TrimEnd('\\', '/').ToLowerInvariant(); }
        catch { return path.ToLowerInvariant(); }
    }

    static string Quote(string a) => a.Length > 0 && !a.Contains(' ') && !a.Contains('"') ? a : "\"" + a.Replace("\"", "\\\"") + "\"";

    /// <summary>后台启动 python -m module args，输出写到 logFile。进程与本程序脱钩：关掉工作台不会停止它。</summary>
    public static int StartDetached(string module, IEnumerable<string> args, string logFile, bool append = false)
    {
        var s = Settings.Current;
        Directory.CreateDirectory(Path.GetDirectoryName(logFile)!);
        var cmdline = $"\"{Quote(s.Python)} -m {module} {string.Join(" ", args.Select(Quote))} {(append ? ">>" : ">")} {Quote(logFile)} 2>&1\"";
        var psi = new ProcessStartInfo("cmd.exe", "/d /s /c " + cmdline)
        {
            WorkingDirectory = s.ProjectDir,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        using var p = Process.Start(psi)!;
        return p.Id;
    }

    public static void Kill(int pid)
    {
        try { using var p = Process.GetProcessById(pid); p.Kill(entireProcessTree: true); } catch { }
    }

    /// <summary>前台运行一个命令，逐行回调输出（用于自检、测速、产数据）。返回退出码。</summary>
    public static async Task<int> RunCaptured(string exe, IEnumerable<string> args, string workDir, Action<string> onLine,
                                              CancellationToken ct = default, IDictionary<string, string>? env = null)
    {
        var psi = new ProcessStartInfo(exe)
        {
            WorkingDirectory = workDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        if (env != null) foreach (var (k, v) in env) psi.Environment[k] = v;
        using var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, e) => { if (e.Data != null) onLine(e.Data); };
        p.ErrorDataReceived += (_, e) => { if (e.Data != null) onLine(e.Data); };
        p.Start();
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        using (ct.Register(() => { try { p.Kill(true); } catch { } }))
            await p.WaitForExitAsync();
        p.WaitForExit();
        return p.ExitCode;
    }

    /// <summary>读日志文件尾部（兼容 UTF-8 与 PowerShell Tee-Object 写出的 UTF-16）。</summary>
    public static string Tail(string file, int maxLines = 300)
    {
        try
        {
            if (!File.Exists(file)) return "";
            using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            var len = fs.Length;
            var take = (int)Math.Min(len, 256 * 1024);
            var head = new byte[2];
            fs.Read(head, 0, 2);
            bool utf16 = head[0] == 0xFF && head[1] == 0xFE;
            fs.Seek(len - take, SeekOrigin.Begin);
            var buf = new byte[take];
            int n = 0;
            while (n < take) { int r = fs.Read(buf, n, take - n); if (r <= 0) break; n += r; }
            var text = (utf16 ? Encoding.Unicode : Encoding.UTF8).GetString(buf, 0, n).TrimStart('\uFEFF');
            var lines = text.Replace("\r\n", "\n").Split('\n');
            var start = Math.Max(0, lines.Length - maxLines);
            if (len > take && start == 0) start = 1;  // 第一行可能被截断
            return string.Join(Environment.NewLine, lines.Skip(start));
        }
        catch { return ""; }
    }

    public static void OpenUrl(string url)
    {
        try { Process.Start(new ProcessStartInfo(url) { UseShellExecute = true }); } catch { }
    }

    public static void OpenFolder(string dir)
    {
        try { if (Directory.Exists(dir)) Process.Start(new ProcessStartInfo("explorer.exe", Quote(dir)) { UseShellExecute = true }); } catch { }
    }
}
