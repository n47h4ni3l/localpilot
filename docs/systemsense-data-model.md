# SystemSense evidence and presentation model

SystemSense has two different consumers with different needs. They must not share the same abstraction boundary.

## Canonical machine truth

LocalPilot reasons from the raw read-only collector outputs. The canonical dynamic snapshot keeps the original `psutil` state, Windows performance/WMI data, and the complete hardware-provider sensor payload with sensor name, type, identifier, hardware identity, value, minimum and maximum. Raw inventory and driver evidence remain available through their existing read-only surfaces.

The desktop summary is never evidence for LocalPilot. When LocalPilot needs a hardware-specific conclusion, diagnosis or action, it should inspect the raw SystemSense surfaces and retain source provenance. The passive prompt context contains raw collector output plus provider availability; the full sensor list remains available through the model-facing SystemSense read tool instead of being injected into every turn.

## Human presentation snapshot

The desktop notepad is deliberately simplified. It derives a small presentation snapshot from the raw truth without modifying or replacing that truth.

Temperature is presented as a component-balanced system average. SystemSense first chooses one representative temperature for CPU, GPU, memory, storage and motherboard classes, then averages the available component representatives. This prevents hardware classes that expose many sensors from dominating the displayed value. Threshold/limit sensors and zero-value placeholders are excluded. The actual peak current temperature is retained separately and drives the thermal-state safety classification.

VRAM is classified by sensor meaning rather than by selecting the largest memory-like value. `SmallData` values are treated as MiB, `Data` values as GiB and converted to MiB. Dedicated used, dedicated free, total and shared-used memory remain distinct. Missing used or total values may be reconstructed only when the complementary explicit fields make the calculation unambiguous.

The presentation payload is marked `presentation_only=true`. Compatibility fields used by the existing desktop UI may map to friendlier values; for example, the legacy temperature card field carries the system-average temperature while the peak remains available separately. Those compatibility fields must not flow back into LocalPilot's reasoning context.

## Rule

The data flow is one-way:

`raw collector truth -> optional derived/history metrics -> user presentation snapshot`

LocalPilot's reasoning path branches from raw collector truth, not from the user presentation snapshot.
