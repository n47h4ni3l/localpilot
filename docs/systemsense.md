# SystemSense environmental awareness

SystemSense is LocalPilot's passive, read-only view of the physical and software
environment supporting it. It samples below the model, stores rolling local
history, derives a small health state, and supplies that compact state to an
operator turn only when the persistent worker has a current sample. Raw data is
not poured into every prompt.

## Runtime shape

```text
Windows / hardware
  ├─ psutil counters (CPU, RAM, swap, disks, network, battery, processes)
  ├─ COM/WMI/CIM (GPU engines/memory, processor limits, thermal zones, power plan)
  ├─ COM/WMI/CIM inventory (board, BIOS, CPU, GPU, DIMMs, disks, NICs, PnP)
  ├─ WMI sensor namespace (LibreHardwareMonitor, then OpenHardwareMonitor)
  └─ PnPUtil CSV (installed third-party driver-store packages)
             │
             ▼
       SystemSense worker
  dynamic sample every 5s by default
  inventory refresh every 15m by default
             │
             ▼
  localpilot-data/systemsense.sqlite3
  snapshots + normalized metrics + inference metrics
             │
       ┌─────┴────────┐
       ▼              ▼
compact derived state bounded read-only drill-down tools
```

The desktop runtime worker owns the sampler lifecycle. `localpilot chat` also
starts it for the lifetime of the interactive process. Collection failures are
recorded as type-only diagnostics and isolated from operator startup. Missing
WMI classes, performance counters, PnPUtil features, or sensor providers reduce
coverage; they do not make the agent unavailable.

`pywin32` is installed only on Windows and gives the collectors native COM/WMI
access without generating PowerShell scripts. PnPUtil is invoked with a fixed
read-only argument vector, `shell=False`, and a timeout. LibreHardwareMonitor is
optional: if its read-only WMI namespace is available, SystemSense consumes its
temperature, fan, load, clock, power, voltage, data and other sensor rows. It
does not configure the monitor, fans, clocks, voltages, firmware, devices, or
drivers.

## Compact state and derived signals

The normal model context contains a bounded current-turn-only object with health,
CPU/GPU/memory/storage pressure, thermal margin, performance limiting, power
state, VRAM when available, device problem counts, inference speed versus its
baseline, a few robust anomalies, and a few candidate causes. Raw sensor rows,
serial numbers, process details and device topology stay out of normal context.

Metrics maintain a rolling median and median absolute deviation over the
configured baseline window. A current value becomes an anomaly only after at
least six samples and a robust deviation threshold. This makes baselines adapt
to the actual machine instead of assuming one universal normal value.

Ollama stream metadata is recorded after each model call:

- tokens per second and total latency;
- approximate time to first generated token (model load plus prompt evaluation);
- prompt/evaluation token counts and context use;
- the contemporaneous normalized environmental metrics.

The correlation query computes bounded Pearson relationships over the configured
window. These are investigative signals only. The API always states that
correlation does not establish causality.

## Inventory and driver semantics

Inventory preserves Windows' identifiers and associations where providers expose
them: PnP device IDs, hardware/compatible IDs, PCI vendor/device/subsystem IDs,
USB vendor/product IDs, class GUIDs, service names, board/BIOS/product identity,
storage firmware, GPU driver/processor identity, driver INF/provider/version/date/signer, and
device/driver/service relationships.

Devices are classified as:

- `present_ok`;
- `problem`;
- `disabled` (Configuration Manager code 22);
- `hidden_or_disconnected` (not present, including code 45).

Drivers and packages are deliberately conservative:

- `active_bound`;
- `installed_bound_inactive`;
- `problematic`;
- `association_unknown`;
- `orphan_candidate_requires_review`;
- `duplicate_older_candidate_requires_review`.

The last two labels never mean "useless" or "safe to delete". A package can be
dormant for disconnected hardware, boot/recovery, rollback, firmware update or
another non-obvious Windows path. Every package record reports
`safe_to_delete: false`; SystemSense has no deletion or driver-management API.

## Query surfaces

All six registered surfaces are `READ_ONLY`, bounded, and expose no raw SQL:

- `get_system_sense_summary()`
- `inspect_hardware_inventory(section, limit)`
- `inspect_driver_inventory(classification, limit)`
- `get_system_sense_history(metric, hours, limit)`
- `get_workload_correlations(limit)`
- `inspect_raw_system_sense(category, limit)`

The intended flow is summary first, one narrow inventory/history query second,
and raw telemetry only when needed to resolve a specific question.

The expanded desktop chat also has a SystemSense quick-glance panel. Its header
indicator refreshes at a low background cadence; opening the panel shows the
latest health state, six core metrics, inference speed versus baseline, current
signals, and bounded background-process pressure. The WebView reads only the
authenticated `GET /v1/systemsense/summary` broker route. The broker reads the
existing telemetry store without starting a collector, and there is no matching
write route or hardware-control action.

## Configuration and privacy

The `[systemsense]` section controls the database name, sample/inventory
intervals, retention, baseline/correlation windows, process row limit, and
compact context injection. Every value is bounded during configuration loading;
the database must be one local filename distinct from chat, learning and library
databases.

The database can contain device serials, local process names and machine-specific
performance history. Treat it as private workstation data. Retention cleanup is
automatic, and SystemSense does not copy observations into `LearningMemory`,
chat history, background-reading notes, self-development records, or GitHub.

## Current coverage limits

Sensor availability is hardware, firmware, driver and provider dependent.
Windows often does not expose fan, VRAM, rail-power or accurate component
temperature data through standard WMI; LibreHardwareMonitor fills many of those
gaps when its WMI interface is enabled. PnPUtil CSV support also varies by
Windows build. Missing fields remain explicit `null`/unavailable values rather
than guessed telemetry.
