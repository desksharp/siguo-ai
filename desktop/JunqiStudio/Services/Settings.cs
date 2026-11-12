using System.IO;
using System.Text.Json;

namespace JunqiStudio.Services;

/// <summary>路径与少量状态，存在 exe 旁边的 settings.json。</summary>
public sealed class Settings
{
    public string ProjectDir { get; set; } = "";
    public string Python { get; set; } = "";
    public string BenchDir { get; set; } = "";
    public int PlayPort { get; set; } = 8765;
    public string PlayModel { get; set; } = "";
    public string LastCheckTime { get; set; } = "";
    public bool? LastCheckOk { get; set; }
    public string LastBench { get; set; } = "";

    static string FilePath => Path.Combine(AppContext.BaseDirectory, "settings.json");
    static readonly JsonSerializerOptions Opt = new() { WriteIndented = true };

    public static Settings Current { get; private set; } = Load();

    static Settings Load()
    {
        Settings s;
        try { s = File.Exists(FilePath) ? JsonSerializer.Deserialize<Settings>(File.ReadAllText(FilePath)) ?? new() : new(); }
        catch { s = new(); }
        // 默认布局：exe 在 <根>\JunqiStudio\，项目在 <根>\junqi-ai，环境在 <根>\venv
        var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, ".."));
        if (string.IsNullOrWhiteSpace(s.ProjectDir)) s.ProjectDir = Path.Combine(root, "junqi-ai");
        if (string.IsNullOrWhiteSpace(s.Python)) s.Python = Path.Combine(root, "venv", "Scripts", "python.exe");
        if (string.IsNullOrWhiteSpace(s.BenchDir)) s.BenchDir = Path.Combine(root, "rocmbench");
        return s;
    }

    public void Save()
    {
        try { File.WriteAllText(FilePath, JsonSerializer.Serialize(this, Opt)); } catch { }
    }

    public string RunsDir => Path.Combine(ProjectDir, "runs");
    public string DataDir => Path.Combine(ProjectDir, "data");
    public string LogsDir => Path.Combine(ProjectDir, "logs");
}
