# Install LocalPilot on Windows

Download the Windows installer from the repository's GitHub Releases and run
it. Approve the Windows administrator prompt. The installer places LocalPilot
in `%LOCALAPPDATA%\LocalPilot\app`, installs missing prerequisites through
WinGet, creates the **LocalPilot** desktop icon and opens the desktop UI.
The ZIP alternative contains **Install LocalPilot.cmd**; extract it and
double-click that file. Windows 10/11 with a 64-bit processor, an internet
connection and Microsoft App Installer/WinGet are required.

The first installation downloads the configured local model, `gpt-oss:20b`
by default. This is a large download and requires sufficient RAM and disk
space to run. Progress and errors are saved in
`%LOCALAPPDATA%\LocalPilot\installation.log`. If a download or prerequisite
installation fails, rerun the installer to continue.

The installer uses an isolated Python environment belonging to LocalPilot.
It installs Python 3.12 and Ollama when absent, plus Git, GitHub CLI,
Microsoft WebView2 and the PawnIO hardware sensor driver. When the configured
implementation backend uses Claude Code, setup installs its native CLI if
missing and checks the selected local model's implementation context. This
local Ollama configuration does not require Anthropic sign-in. A custom Claude
executable path or a disabled implementation backend is preserved. The release bundle
already contains the SystemSense hardware helper, so a release installation
does not require the .NET SDK. A Git source checkout builds the helper if it
is missing and installs the .NET SDK when needed.

Existing `localpilot.toml`, model selection, local data, Git history and Python
environment are preserved on reinstallation. An existing installation keeps
its checkout rather than having bundle files copied over it; use the desktop
update flow to apply a newer revision. A custom local `nestra:*` model is never
replaced or downloaded from a public registry if it is missing.

The desktop icon requests administrator access each time it launches, allowing
the hardware provider to access CPU sensors. Windows still requires UAC
approval. If the PawnIO installation requests a Windows restart, restart the
PC and then open LocalPilot using its desktop icon.

LocalPilot's source checkout is connected to its GitHub repository for updates
and repair pull requests. Desktop chat does not require GitHub sign-in.
To enable GitHub work, complete `gh auth login --web` once and configure your
Git author name and email if this PC has never been used with Git. The
installer does not collect credentials or change your global Git identity.

For an existing Git checkout, double-click **Install LocalPilot.cmd** there to
install in place. This leaves the checkout's configuration and data intact.
Advanced installations can call `scripts/install-localpilot.ps1` with
`-InstallDirectory <directory>` and `-SkipLaunch`. `-ShortcutPath <path>`
creates the shortcut at an alternative path for a separate installation.
