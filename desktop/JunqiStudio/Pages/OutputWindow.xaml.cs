using System.Text;
using System.Windows;
using System.Windows.Threading;
using JunqiStudio.Services;

namespace JunqiStudio.Pages;

/// <summary>运行一个前台命令并实时显示输出（自检、测速、产数据）。</summary>
public partial class OutputWindow : Window
{
    readonly CancellationTokenSource _cts = new();
    readonly StringBuilder _pending = new();
    readonly DispatcherTimer _flush;
    public List<string> Lines { get; } = new();
    public int? ExitCode { get; private set; }

    public OutputWindow(string title, string sub)
    {
        InitializeComponent();
        Owner = Application.Current.MainWindow;
        Title = title;
        TitleText.Text = title;
        SubText.Text = sub;
        Ui.Badge(StateBadge, StateText, "运行中", "amber");
        _flush = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(150) };
        _flush.Tick += (_, _) => Flush();
        _flush.Start();
        Closing += (_, e) =>
        {
            if (ExitCode == null && !Ui.Confirm("任务还在运行，关闭窗口会停止它。确定关闭？")) { e.Cancel = true; return; }
            _cts.Cancel();
        };
    }

    void Flush()
    {
        string add;
        lock (_pending) { if (_pending.Length == 0) return; add = _pending.ToString(); _pending.Clear(); }
        Out.AppendText(add);
        Out.ScrollToEnd();
    }

    public async Task<int> Run(string exe, IEnumerable<string> args, string workDir, Func<int, string>? describe = null,
                               IDictionary<string, string>? env = null)
    {
        Show();
        int code;
        try
        {
            code = await Procs.RunCaptured(exe, args, workDir, line =>
            {
                lock (_pending) { _pending.AppendLine(line); }
                lock (Lines) Lines.Add(line);
            }, _cts.Token, env);
        }
        catch (Exception ex)
        {
            lock (_pending) _pending.AppendLine("启动失败：" + ex.Message);
            code = -1;
        }
        ExitCode = code;
        Flush();
        _flush.Stop();
        StopBtn.IsEnabled = false;
        var ok = code == 0;
        Ui.Badge(StateBadge, StateText, _cts.IsCancellationRequested ? "已停止" : ok ? "完成" : $"失败（{code}）",
                 _cts.IsCancellationRequested ? "grey" : ok ? "green" : "red");
        if (describe != null) SubText.Text = describe(code);
        return code;
    }

    Action? _action;

    /// <summary>任务完成后在底部显示一个主按钮（例如“重启并更新”）。</summary>
    public void ShowAction(string text, Action onClick)
    {
        _action = onClick;
        ActionBtn.Content = text;
        ActionBtn.Visibility = Visibility.Visible;
    }

    void Action_Click(object sender, RoutedEventArgs e) => _action?.Invoke();
    void Stop_Click(object sender, RoutedEventArgs e) => _cts.Cancel();
    void Close_Click(object sender, RoutedEventArgs e) => Close();
}
