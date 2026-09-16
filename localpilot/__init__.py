__version__ = "0.2.0"

# SystemSense prefers LocalPilot's bundled read-only hardware provider while
# preserving the existing Libre/OpenHardwareMonitor WMI fallbacks. Installing
# the provider here keeps every SystemSense entry point on the same sensor
# source without duplicating selection logic across the UI, tools and backend.
from localpilot.systemsense_hardware import install_bundled_hardware_provider as _install_hardware_provider

_install_hardware_provider()
del _install_hardware_provider
