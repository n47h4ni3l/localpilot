# Windows release signing

LocalPilot's public Windows installer requires Authenticode signatures from a
validated publisher. The existing 0.2.1 draft assets were built unsigned;
they must be replaced with verified signed assets before public distribution.
SHA-256 checksums identify bytes but do not establish a trusted publisher.

The signing account, legal identity validation and private key are external
prerequisites. LocalPilot does not generate a development certificate, import
a root certificate, or disable Windows security to simulate public trust.
Use a publicly trusted certificate authority and its hardware token or cloud
HSM. Keep the private key and credentials outside the repository.

## Choose the publisher

Microsoft's current [Artifact Signing quickstart](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart)
includes Australian organizations for Public Trust. Individual developers are
currently limited to the US and Canada. An Australian legal organization or
DBA must satisfy the service's business and representative identity validation;
having a GitHub account alone is not enough. Some older Microsoft overview
pages still list fewer eligible countries, so use the service quickstart and
confirm eligibility during onboarding.

For personal publishing, a CA offering individual validation and cloud signing,
such as [SSL.com IV](https://www.ssl.com/products/software-integrity/code-signing/iv/),
is a candidate. Confirm Australian validation eligibility, publisher name and
all certificate/cloud-signing fees before purchasing a service.

The [SignPath Foundation programme](https://signpath.org/terms.html) requires an
OSI-approved open-source licence. LocalPilot's current source-visible licence
does not meet that requirement; obtaining free signing must not silently change
the project's licence.

Signing establishes identity and integrity. It does not guarantee immediate
SmartScreen reputation or remove Windows UAC approval. See Microsoft's
[SmartScreen guidance](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation).
The installer and its dependencies still require real installation testing on
a fresh Windows PC, including Smart App Control where enabled.

## Build a signed package

After the provider is configured, supply an authenticated PowerShell adapter
that accepts a `-Paths` string array. It must sign every supplied file with the
chosen public certificate, SHA-256 digest and an RFC 3161 timestamp, throw on failures, and
return a failing exit status if its signing client fails. The adapter runs only
on the maintainer's build machine or protected CI runner; it is never needed on
an end user's PC. It must not modify certificate trust stores.

Before creating the release commit, use that adapter to sign the tracked
PowerShell scripts, verify their signatures, then review and commit them.
The `*.ps1 -text` Git attribute preserves the exact signed bytes. Any subsequent
script edit needs a new signature before release. The builder verifies the
committed copies without adding signature blocks: changing them inside the
package would leave a fresh Git installation dirty and break its worker and
update checks. Keep the provider adapter outside released source.

```powershell
./scripts/build-windows-installer.ps1 `
    -SigningScriptPath <authenticated-provider-adapter.ps1> `
    -ExpectedPublisher '<exact full certificate Subject>'
```

`-SignToolPath` can select a Windows SDK SignTool explicitly. Otherwise setup
locates an installed SDK tool. Verification requires embedded Authenticode
signatures, a valid code-signing certificate matching the exact expected
Subject, a timestamp, and successful `signtool verify /pa /all /tw /v` without
warnings. Use a clean signing runner with standard Windows public trust roots;
do not import a private test root to make verification pass.

The builder verifies the already-signed committed scripts and signs the generated
sensor helper before computing the helper hash and creating the ZIP. It signs the extraction script before
embedding the wrapper, then signs the final EXE. It checks extraction using that
signed EXE and computes download checksums last. Any failed signing or
verification step prevents the build being reported ready. Partial output from
a failed build must not be published.

For unsigned development validation only:

```powershell
./scripts/build-windows-installer.ps1 -AllowUnsignedPreview
```

These files use the `LocalPilot-Preview` prefix and declare
`release_channel: unsigned-preview` in their manifest. They must not be promoted
to public installers by renaming them.

## GitHub release workflow

The workflow-dispatch default builds an unsigned preview for testing. A signed
run and a published-release run require `WINDOWS_SIGNING_SCRIPT` and
`WINDOWS_SIGNING_PUBLISHER` repository variables; the first identifies the
reviewed adapter path, and the second is the validated certificate Subject.
Configure provider authentication before the build step using the provider's
documented integration. Protect the release environment and credentials, and
grant only the signing permissions it requires. The provider-specific login
integration cannot be completed until a signing account exists.

The public-release build has no unsigned fallback. It fails if the signing
adapter/identity is missing or any signature cannot be verified. The workflow's
`release: published` trigger runs after GitHub publication; it cannot prevent a
human from publishing an existing draft with old assets. Review and replace all
unsigned draft assets with the verified signed EXE, ZIP and final checksums
before publishing. A successful unsigned-preview build is not release approval.
