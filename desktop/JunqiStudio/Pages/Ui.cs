using System.Windows.Shapes;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace JunqiStudio.Pages;

public static class Ui
{
    public static Brush Res(string key) => (Brush)Application.Current.FindResource(key);

    /// <summary>徽章配色：green 运行中/通过，amber 进行中/提醒，red 失败，primary 完成，grey 空闲。</summary>
    public static void Badge(Border b, TextBlock t, string text, string kind)
    {
        t.Text = text;
        (b.Background, t.Foreground) = kind switch
        {
            "green" => (Res("GreenSoft"), Res("Green")),
            "amber" => (Res("AmberSoft"), Res("Amber")),
            "red" => (Res("RedSoft"), Res("Red")),
            "primary" => (Res("PrimarySoft"), Res("Primary")),
            _ => (Res("GreySoft"), Res("Text2")),
        };
    }

    /// <summary>左侧大圆点：外圈浅色、内点实色。</summary>
    public static void Dot(Ellipse ring, Ellipse dot, string kind)
    {
        (ring.Fill, dot.Fill) = kind switch
        {
            "green" => (Res("GreenRing"), Res("Green")),
            "amber" => (Res("AmberSoft"), Res("Amber")),
            "red" => (Res("RedSoft"), Res("Red")),
            "primary" => (Res("PrimarySoft"), Res("Primary")),
            _ => (Res("GreySoft"), Res("Grey")),
        };
    }

    public static string Duration(double seconds)
    {
        if (double.IsNaN(seconds) || seconds < 0) return "-";
        if (seconds < 60) return $"{seconds:0} 秒";
        if (seconds < 3600) return $"{seconds / 60:0} 分钟";
        return $"{(int)(seconds / 3600)} 小时 {(int)(seconds % 3600 / 60)} 分钟";
    }

    public static bool Confirm(string text, string title = "确认") =>
        MessageBox.Show(Window.GetWindow(Application.Current.MainWindow), text, title, MessageBoxButton.OKCancel, MessageBoxImage.Question) == MessageBoxResult.OK;

    public static void Info(string text, string title = "提示") =>
        MessageBox.Show(Window.GetWindow(Application.Current.MainWindow), text, title, MessageBoxButton.OK, MessageBoxImage.Information);
}
