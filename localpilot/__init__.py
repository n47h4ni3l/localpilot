__version__ = "0.2.0"

# SystemSense prefers LocalPilot's bundled read-only hardware provider while
# preserving the existing Libre/OpenHardwareMonitor WMI fallbacks. Installing
# the provider here keeps every SystemSense entry point on the same sensor
# source without duplicating selection logic across the UI, tools and backend.
from localpilot.systemsense_hardware import install_bundled_hardware_provider as _install_hardware_provider

_install_hardware_provider()
del _install_hardware_provider

# Keep LocalPilot's model-facing machine evidence on canonical raw collector
# output while deriving a separate, one-way normalized snapshot for the human
# desktop UI. The presentation layer must never become model evidence.
from localpilot.systemsense_views import install_systemsense_truth_views as _install_systemsense_truth_views

_install_systemsense_truth_views()
del _install_systemsense_truth_views

# Baseline deviations are useful context, but a machine being busier than its
# usual idle baseline is not automatically unhealthy. Apply conservative
# health semantics only to the human presentation summary; raw model evidence
# remains untouched.
from localpilot.systemsense_presentation_health import (
    install_systemsense_presentation_health as _install_systemsense_presentation_health,
)

_install_systemsense_presentation_health()
del _install_systemsense_presentation_health
