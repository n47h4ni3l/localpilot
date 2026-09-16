using System.Text.Json;
using LibreHardwareMonitor.Hardware;

namespace LocalPilot.SystemSense.HardwareProvider;

internal sealed class UpdateVisitor : IVisitor
{
    public void VisitComputer(IComputer computer) => computer.Traverse(this);

    public void VisitHardware(IHardware hardware)
    {
        hardware.Update();
        foreach (IHardware subHardware in hardware.SubHardware)
        {
            subHardware.Accept(this);
        }
    }

    public void VisitSensor(ISensor sensor) { }

    public void VisitParameter(IParameter parameter) { }
}

internal static class Program
{
    private const string SourceName = "LibreHardwareMonitorLib";
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
    };

    public static int Main(string[] args)
    {
        if (!OperatingSystem.IsWindows())
        {
            WriteError("platform", "The bundled hardware provider is available on Windows only.");
            return 2;
        }

        Computer computer = new()
        {
            IsCpuEnabled = true,
            IsGpuEnabled = true,
            IsMemoryEnabled = true,
            IsMotherboardEnabled = true,
            IsControllerEnabled = true,
            IsNetworkEnabled = true,
            IsStorageEnabled = true,
        };

        try
        {
            computer.Open();

            if (args.Any(arg => string.Equals(arg, "--snapshot", StringComparison.OrdinalIgnoreCase)))
            {
                WriteSnapshot(computer);
                return 0;
            }

            return RunStdio(computer);
        }
        catch (Exception exc)
        {
            WriteError("startup", exc.GetType().Name);
            return 1;
        }
        finally
        {
            computer.Close();
        }
    }

    private static int RunStdio(Computer computer)
    {
        string? line;
        while ((line = Console.ReadLine()) is not null)
        {
            string command = line.Trim();
            if (command.Length == 0)
            {
                continue;
            }

            if (string.Equals(command, "quit", StringComparison.OrdinalIgnoreCase))
            {
                return 0;
            }

            if (string.Equals(command, "ping", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    ok = true,
                    source = SourceName,
                    command = "pong",
                }, JsonOptions));
                Console.Out.Flush();
                continue;
            }

            if (!string.Equals(command, "snapshot", StringComparison.OrdinalIgnoreCase))
            {
                WriteError("command", "unsupported");
                continue;
            }

            try
            {
                WriteSnapshot(computer);
            }
            catch (Exception exc)
            {
                WriteError("snapshot", exc.GetType().Name);
            }
        }

        return 0;
    }

    private static void WriteSnapshot(Computer computer)
    {
        computer.Accept(new UpdateVisitor());
        List<object> sensors = [];

        foreach (IHardware hardware in computer.Hardware)
        {
            AddHardwareSensors(hardware, sensors);
        }

        string? version = typeof(Computer).Assembly.GetName().Version?.ToString();
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            ok = true,
            source = SourceName,
            provider_version = version,
            sensors,
            errors = Array.Empty<string>(),
        }, JsonOptions));
        Console.Out.Flush();
    }

    private static void AddHardwareSensors(IHardware hardware, List<object> output)
    {
        foreach (ISensor sensor in hardware.Sensors)
        {
            output.Add(new
            {
                Identifier = sensor.Identifier.ToString(),
                Name = sensor.Name,
                SensorType = sensor.SensorType.ToString(),
                Value = Finite(sensor.Value),
                Min = Finite(sensor.Min),
                Max = Finite(sensor.Max),
                Parent = hardware.Identifier.ToString(),
                HardwareName = hardware.Name,
                HardwareType = hardware.HardwareType.ToString(),
            });
        }

        foreach (IHardware subHardware in hardware.SubHardware)
        {
            AddHardwareSensors(subHardware, output);
        }
    }

    private static double? Finite(float? value)
    {
        if (!value.HasValue || float.IsNaN(value.Value) || float.IsInfinity(value.Value))
        {
            return null;
        }

        return value.Value;
    }

    private static void WriteError(string stage, string error)
    {
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            ok = false,
            source = SourceName,
            stage,
            error,
            sensors = Array.Empty<object>(),
        }, JsonOptions));
        Console.Out.Flush();
    }
}
