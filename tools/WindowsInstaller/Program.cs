using System.Diagnostics;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

internal static class Program
{
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBox(IntPtr window, string text, string caption, uint type);

    private static int Main(string[] arguments)
    {
        bool extractOnly = arguments.Length == 1 &&
            (arguments[0].StartsWith("/extract:", StringComparison.OrdinalIgnoreCase) ||
             arguments[0].StartsWith("--extract:", StringComparison.OrdinalIgnoreCase));
        string? extraction = null;
        try
        {
            if (arguments.Length != 0 && !extractOnly)
                throw new ArgumentException("Use the installer without arguments, or /extract:<directory> to inspect its contents without installing.");
            extraction = extractOnly
                ? Path.GetFullPath(arguments[0][(arguments[0].IndexOf(':') + 1)..])
                : Path.Combine(Path.GetTempPath(), "LocalPilotBootstrap-" + Guid.NewGuid().ToString("N"));
            using Stream payload = Assembly.GetExecutingAssembly().GetManifestResourceStream("LocalPilot.InstallerPayload")
                ?? throw new InvalidOperationException("The installer has no embedded release payload.");
            Directory.CreateDirectory(extraction);
            ZipFile.ExtractToDirectory(payload, extraction);
            if (extractOnly)
                return 0;

            string setup = Path.Combine(extraction, "Extract-Setup.ps1");
            if (!File.Exists(setup))
                throw new InvalidOperationException("The extracted installer entrypoint is missing.");
            string powershell = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),
                "WindowsPowerShell", "v1.0", "powershell.exe");
            var start = new ProcessStartInfo(powershell)
            {
                Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File \"" + setup + "\"",
                WorkingDirectory = extraction,
                UseShellExecute = true,
                WindowStyle = ProcessWindowStyle.Normal,
            };
            using Process process = Process.Start(start) ?? throw new InvalidOperationException("Windows did not start setup.");
            process.WaitForExit();
            if (process.ExitCode != 0)
                throw new InvalidOperationException($"Setup exited with code {process.ExitCode}. Its extracted evidence is preserved at {extraction}.");
            CleanupSuccessfulExtraction(extraction);
            return 0;
        }
        catch (Exception error)
        {
            string log = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LocalPilot", "installation.log");
            string message = error.Message + Environment.NewLine + Environment.NewLine + "Setup log: " + log;
            if (extractOnly)
                Console.Error.WriteLine(message);
            else
                MessageBox(IntPtr.Zero, message, "LocalPilot setup did not finish", 0x00000010);
            return 1;
        }
    }

    private static void CleanupSuccessfulExtraction(string directory)
    {
        string resolved = Path.GetFullPath(directory);
        string prefix = Path.GetFullPath(Path.GetTempPath()).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        if (!resolved.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) ||
            !Regex.IsMatch(Path.GetFileName(resolved), "^LocalPilotBootstrap-[0-9a-f]{32}$"))
            throw new InvalidOperationException("Refusing to clean an unexpected installer extraction directory.");
        Directory.Delete(resolved, recursive: true);
    }
}
