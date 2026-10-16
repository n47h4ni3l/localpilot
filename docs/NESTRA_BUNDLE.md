# Nestra ALPHA2 private bundle candidate

This **0.2.2-alpha.2** bundle is a local, private Windows evaluation candidate containing the accepted **Nestra P1.2** model, its desktop application, and the pinned runtimes needed for an offline installation. It uses native Ollama implementation through `local_tools`; a Claude Code executable is neither included nor required. Public redistribution review remains incomplete for the project source and the selected components. Read `THIRD_PARTY_NOTICES.md` before sharing or publishing an archive.

The package preserves the P1.2 model identity, temperature 1.0, and guide-only behavior. Chat, planning, research, implementation, review, and repair use the same accepted `nestra:p1.2` model. The fresh configuration explicitly selects `implementation_backend = "local_tools"`, keeps both developer and implementation model selectors on P1.2, and leaves developer-model fallbacks empty. Native implementation uses the existing confined file tools and retains the resource, candidate-validation, checkpoint, review, CI, and human-merge gates. It does not train a model, promote a new checkpoint, or establish a quality improvement. The source includes the chat, Files, Activity, and library interfaces. The presence of those interfaces and successful packaging checks do not establish that they have passed acceptance testing on a fresh Windows PC. A clean Windows VM installation and desktop acceptance run remain separate verification gates.

## PC and storage requirements

- Windows x64, Windows 10 build 19041 or newer, and an AVX2-capable x64 processor. ARM Windows is not supported by this bundle.
- At least **24 GiB of physical RAM**; **32 GiB or more is recommended**. Model context, other applications, and GPU offload affect memory use and speed. This is a minimum preflight requirement, not a performance guarantee.
- A local fixed drive with sufficient space for the downloaded ZIP, its fully extracted contents, and the installed copy. Model weights alone are approximately **13.8 GB**, before Python, Ollama, PortableGit, other runtimes, and installation files. The exact final archive size is recorded in the build receipt.
- Setup checks free space against **twice the uncompressed bundle inventory plus 5 GiB**. Keep additional space for conversations, files, model context, logs, and future work.

Setup does not change Windows paging, antivirus, driver security, or other security settings to make an unsupported PC pass these checks.

## Install the complete extracted bundle

1. Keep the ZIP and its accompanying SHA256/build receipt together. Verify the archive checksum supplied with this particular candidate.
2. Use **Extract All** to extract the entire ZIP to a local folder. Do not run setup from inside the ZIP or copy only the installer command file.
3. Open **Install Nestra.cmd** in the extracted folder.
4. Accept the Windows administrator prompt using the **same Windows account** that opened setup. Using a different administrator account is rejected so that the application, model store, and login tasks remain owned by the intended user.
5. Allow setup to verify all inventoried files and install the private runtime. Follow any restart instruction in the final setup message before opening Nestra.

The default destination is `%LOCALAPPDATA%\Programs\Nestra`. This candidate supports **fresh installation only**. If the destination, its shortcut, or its scheduled tasks already exist, setup stops and preserves them. A failed partial installation is retained for diagnosis. An upgrade or migration requires a separate reviewed procedure; setup does not overwrite an existing configuration, conversation database, library, or draft.

Advanced evaluation options are available on `scripts\install-nestra-bundle.ps1`: `-InstallDirectory`, `-VerifyOnly`, `-SkipLaunch`, and `-SkipHardwareInstall`. `-VerifyOnly` hashes the inventory and performs local platform, capacity, and launcher-conflict checks. It does not run a bundled executable, create an installation directory or receipt, or register tasks. Its AVX2 check uses a Windows system API; Windows PowerShell may create temporary compiler scratch files while preparing that probe. The normal installer requires the owner SID supplied by `Install Nestra.cmd`.

## What setup creates

The installation has separate `app`, `runtime`, `models`, `library`, and `state` directories. It creates a fresh application configuration and local data directory. The accepted model's original Ollama manifest and blobs are copied into this installation's model store and checked against their recorded hashes; no substitute model is downloaded or selected.

Python and Tk, Ollama, PortableGit with Bash, GitHub CLI, and the wheelhouse are private runtime payloads. Evolution implementation runs through Nestra's native local tools instead of an external Claude Code process. Python dependencies install from the hash-pinned wheelhouse without contacting a package index. Git source is restored from the included `source.bundle` at the exact recorded `main` commit, without fetching from the internet. Component versions and hashes are recorded in `packaging/nestra-alpha2-dependencies.json` in the source and in the assembled bundle receipts.

Setup may install the bundled **Microsoft WebView2 Runtime** when absent, and the bundled **PawnIO hardware driver** when absent and hardware installation has not been skipped. These are Windows prerequisites installed by their own upstream installers. They are distinct from the private Python and model directories.

The desktop runs under a **Limited** same-account login task. The broker and background worker run under **Highest** same-account login tasks so that the existing hardware access model is retained. Setup records the per-install task names in `tasks.json` and creates the Nestra ALPHA2 desktop shortcut. The first desktop launch goes through Explorer and the Limited task. No global PATH modification or Git account/identity configuration is performed.

The desktop's Start with Windows setting reports the verified state of the three bundle tasks and is read-only. Its explanation names the exact tasks to inspect in Windows Task Scheduler as administrator under this account. In-app updates are unavailable for this fresh-install-only candidate; a future bundle migration procedure is required. These controls do not register a separate legacy LocalPilot startup task or hand the bundle to the legacy updater.

## Isolation and offline use

The bundle reserves loopback **11435** for its private Ollama runtime and **8766** for its broker. It does not take over an existing Ollama server on 11434 or an existing LocalPilot broker on 8765. An unexpected process on one of the bundle's reserved ports is an error; the launcher must stop rather than replace that process or silently connect to a different model. A second bundle installation cannot share occupied reserved ports merely because it has a different folder.

Once a complete candidate is installed, model inference and local chat use the included model and local runtimes. Files and activity continue to use the application's existing permission and tool boundaries. Offline packaging does not make every tool or development task an offline operation.

**Online research, remote repositories, GitHub operations, and publishing changes require network access.** GitHub integration and automatic candidate pushing are deliberately disabled in the fresh configuration until the recipient configures their own account and an appropriate repository. Bundling Git and GitHub CLI does not supply repository permissions, login credentials, a Git identity, or authorization to publish. Selecting native implementation does not bypass those account and authorization requirements. Owner approval and existing review/promotion rules still apply.

## Data, diagnostics, and current verification limits

The builder selects committed project source, explicit dependency payloads, the accepted model export, and the hardware provider. It does not copy the owner's running `localpilot-data` directory, personal conversation database, unsent drafts, authentication files, private evaluation inputs, or live memory databases. The source bundle contains committed public Git history, including training lineage and deployment receipts; those records can contain historical machine paths and artifact hashes. They are different from a copy of the owner's personal runtime databases. Source-history review remains part of candidate preparation.

Installation receipts and stage output are kept under `%LOCALAPPDATA%\Nestra\InstallerLogs`. Failures are recorded and partial files are retained. If task registration partly succeeds, the receipt lists each registered task so that partial setup is visible. An `installed` receipt describes completed file/prerequisite setup; it does not prove model quality or fresh-machine desktop acceptance.

This private ALPHA2 candidate has source and preflight checks. It must not be described as installed and accepted on a clean Windows VM unless that separate test has actually completed. Keep the exact archive, manifest, checksums, setup receipt, and any failure logs when reporting results.

Identify this native candidate by its archive SHA256 and manifest `source_sha`; the display version alone does not identify its contents. An earlier private archive with the same display version may contain the previous Claude-based implementation payload. That earlier archive remains separate historical evidence and is not modified by preparing this candidate.

## Licenses and sharing

LocalPilot is governed by its **Source-Visible License**, supplied as `LICENSE.md` in the source. That license permits specified private evaluation and reserves copying, redistribution, derivative-work, commercial, and training rights as described in its text. Source visibility is not a blanket open-source license.

The gpt-oss base model and each runtime have their own terms. Preserve the upstream license and notice files in the payloads, the top-level `licenses` directory, the hardware provider's notices, and the installed source's `packaging/licenses` directory. `THIRD_PARTY_NOTICES.md` records the known license sources and remaining redistribution review. The private candidate label does not itself grant any missing third-party rights.
