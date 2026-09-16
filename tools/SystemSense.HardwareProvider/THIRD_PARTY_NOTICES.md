# SystemSense hardware provider third-party notices

LocalPilot's Windows SystemSense hardware provider uses **LibreHardwareMonitorLib 0.9.6**.

- Project: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor
- Release pinned by LocalPilot: `v0.9.6`
- NuGet package: `LibreHardwareMonitorLib` version `0.9.6`
- License: Mozilla Public License 2.0 (MPL-2.0)
- Upstream license text: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/blob/v0.9.6/LICENSE
- Upstream third-party notices: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/blob/v0.9.6/THIRD-PARTY-NOTICES.txt

LocalPilot does not modify LibreHardwareMonitorLib in this integration. It consumes the published library through a small read-only helper process and exposes the resulting sensor observations to SystemSense. The helper intentionally provides no hardware-control commands.

Packaged LocalPilot distributions that contain the published helper must include this notice and comply with the applicable LibreHardwareMonitor and transitive-dependency license requirements.
