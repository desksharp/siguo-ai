using System.IO;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using JunqiStudio.Services;

namespace JunqiStudio.Pages;

public partial class DataPage : UserControl, IPage
{
    bool _ready;

    public DataPage()
    {
        InitializeComponent();
        Loaded += (_, _) => OnShown();
    }

    public void OnShown()
    {
        var sets = DataSet.LoadAll();
        Grid.ItemsSource = sets;
        _ready = false;
        SeedBox.Text = (sets.Count + 1).ToString();
        _ready = true;
        Form_Changed(this, null!);
    }

    string Rules => (RulesBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "siguo_standard";

    void Form_Changed(object sender, RoutedEventArgs e)
    {
        if (!_ready || OutBox == null) return;
        OutBox.Text = $"data/v2_rule/{(Rules == "duel" ? "duel" : "four")}_s{SeedBox.Text.Trim()}";
    }

    async void Gen_Click(object sender, RoutedEventArgs e)
    {
        var s = Settings.Current;
        var outDir = OutBox.Text.Trim().Replace('\\', '/');
        GenError.Text = "";
        if (!int.TryParse(GamesBox.Text, out var games) || games < 100) { GenError.Text = "局数至少 100。"; return; }
        if (!int.TryParse(SeedBox.Text, out var seed)) { GenError.Text = "随机种子要是整数。"; return; }
        if (!Regex.IsMatch(outDir, @"^data/[A-Za-z0-9_\-/\.]+$")) { GenError.Text = "输出目录要在 data/ 下，只用英文字母、数字和下划线。"; return; }
        if (Directory.Exists(Path.Combine(s.ProjectDir, outDir))) { GenError.Text = "这个目录已经存在，换个名字。"; return; }
        var w = new OutputWindow("产自对弈数据", $"{games:N0} 局 · 随机种子 {seed} → {outDir}");
        await w.Run(s.Python, new[] { "-m", "junqi.gen_data", "--rules", Rules, "--games", games.ToString(), "--seed", seed.ToString(), "--out", outDir },
                    s.ProjectDir, c => c == 0 ? $"完成，数据在 {outDir}。新建训练时可以把它勾上。" : "失败，看上面的输出");
        OnShown();
    }

    void OpenData_Click(object sender, RoutedEventArgs e) => Procs.OpenFolder(Settings.Current.DataDir);
}
