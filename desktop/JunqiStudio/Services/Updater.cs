using System.Diagnostics;
using System.IO;
using System.Text;

namespace JunqiStudio.Services;

/// <summary>程序自更新：项目里的源码（desktop\JunqiStudio）比当前程序新时，重新编译到临时目录，
/// 退出后由一个小脚本把新文件拷过来并重新打开。settings.json 与 webview2 数据不受影响。</summary>
public static class Updater
{
    public static string SourceDir => Path.Combine(Settings.Current.ProjectDir, "desktop", "JunqiStudio");
    static string AppDir => AppContext.BaseDirectory.TrimEnd('\\', '/');
    static string StageDir => AppDir + ".new";

    static readonly string[] SourceExt = { ".cs", ".xaml", ".csproj", ".ico" };

    /// <summary>当前程序对应的源码时间：自更新时写在 build.stamp 里；没有就用主程序集的写入时间。
    /// （增量编译可能产出与旧版一模一样的程序集，文件时间不变，所以不能只看程序集时间。）</summary>
    public static DateTime BuildTime
    {
        get
        {
            var stamp = Path.Combine(AppDir, "build.stamp");
            var dll = File.GetLastWriteTime(Path.Combine(AppDir, "JunqiStudio.dll"));
            try
            {
                if (File.Exists(stamp) && DateTime.TryParse(File.ReadAllText(stamp).Trim(), null,
                        System.Globalization.DateTimeStyles.RoundtripKind, out var t) && t > dll)
                    return t;
            }
            catch { }
            return dll;
        }
    }

    /// <summary>源码最后修改时间；找不到源码返回 null。</summary>
    public static DateTime? SourceTime()
    {
        try
        {
            if (!Directory.Exists(SourceDir)) return null;
            var sep = Path.DirectorySeparatorChar;
            return Directory.EnumerateFiles(SourceDir, "*", SearchOption.AllDirectories)
                .Where(f => SourceExt.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .Where(f => !f.Contains($"{sep}bin{sep}") && !f.Contains($"{sep}obj{sep}"))
                .Select(File.GetLastWriteTime)
                .DefaultIfEmpty(DateTime.MinValue)
                .Max();
        }
        catch { return null; }
    }

    public static bool Available => SourceTime() is { } t && t > BuildTime.AddSeconds(5);

    public static string? DotnetPath()
    {
        var pf = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var p = Path.Combine(pf, "dotnet", "dotnet.exe");
        if (File.Exists(p)) return p;
        foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';'))
            if (dir.Length > 0 && File.Exists(Path.Combine(dir, "dotnet.exe"))) return Path.Combine(dir, "dotnet.exe");
        return null;
    }

    /// <summary>编译到 JunqiStudio.new；成功后返回 true，调用方再调用 <see cref="SwapAndRestart"/>。</summary>
    public static async Task<bool> Build(Pages.OutputWindow w)
    {
        var dotnet = DotnetPath();
        if (dotnet == null) return false;
        if (Directory.Exists(StageDir)) Directory.Delete(StageDir, true);
        var srcTime = SourceTime() ?? DateTime.Now;
        var code = await w.Run(dotnet,
            new[] { "publish", Path.Combine(SourceDir, "JunqiStudio.csproj"), "-c", "Release", "-o", StageDir, "--nologo" },
            SourceDir,
            c => c == 0 ? "编译成功，点“重启并更新”换上新版本（训练和对弈服务不受影响）" : "编译失败，看上面的输出；当前程序没有被改动",
            new Dictionary<string, string> { ["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1", ["DOTNET_NOLOGO"] = "1", ["DOTNET_CLI_UI_LANGUAGE"] = "zh-Hans" });
        if (code != 0 || !File.Exists(Path.Combine(StageDir, "JunqiStudio.exe"))) return false;
        File.WriteAllText(Path.Combine(StageDir, "build.stamp"), srcTime.ToString("o"));
        return true;
    }

    /// <summary>写一个等待本进程退出 → 拷贝新文件 → 重新打开的脚本，后台运行它，然后退出程序。</summary>
    public static void SwapAndRestart()
    {
        var pid = Environment.ProcessId;
        var script = Path.Combine(Path.GetTempPath(), "junqi_studio_update.cmd");
        var sb = new StringBuilder();
        sb.AppendLine("@echo off");
        sb.AppendLine(":wait");
        sb.AppendLine($"tasklist /fi \"PID eq {pid}\" /nh | find \"{pid}\" >nul && (timeout /t 1 /nobreak >nul & goto wait)");
        sb.AppendLine($"robocopy \"{StageDir}\" \"{AppDir}\" /E /R:5 /W:1 /NFL /NDL /NJH /NJS /NP >nul");
        sb.AppendLine($"rmdir /s /q \"{StageDir}\"");
        sb.AppendLine($"start \"\" \"{Path.Combine(AppDir, "JunqiStudio.exe")}\"");
        sb.AppendLine("del \"%~f0\"");
        File.WriteAllText(script, sb.ToString(), Encoding.Default);
        Process.Start(new ProcessStartInfo("cmd.exe", $"/d /c \"{script}\"") { UseShellExecute = false, CreateNoWindow = true });
        System.Windows.Application.Current.Shutdown();
    }
}
