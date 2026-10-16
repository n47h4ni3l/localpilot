# Reviewed source-only dependency

`proxy_tools-0.1.0-py3-none-any.whl` is a pure Python wheel needed by pywebview
6.2.1. PyPI publishes this version as source only. It is prebuilt so recipients
do not execute a source build or need a compiler/network during installation.

- Source: https://files.pythonhosted.org/packages/f2/cf/77d3e19b7fabd03895caca7857ef51e4c409e0ca6b37ee6e9f7daa50b642/proxy_tools-0.1.0.tar.gz
- Source SHA256: `ccb3751f529c047e2d8a58440d86b205303cf0fe8146f784d1cbcd94f0a28010`
- Source size: 2978 bytes. Version metadata checked against https://pypi.org/pypi/proxy_tools/0.1.0/json.
- Built with CPython 3.12 and setuptools 81.0.0 using `pip wheel --no-deps --no-build-isolation`.
- Wheel SHA256: `cf5dfb5581a7ec6745c448d81dcfe6f817490568d7de469dc8760181ef6fe63d`.
- The archive's setup.py, package source and member paths were inspected before
  building. No project owner data is involved. Source bytes are unmodified.
- Upstream metadata declares MIT; the source header describes BSD and credits
  Armin Ronacher and Jonathan Tushman. Preserve that header. The source archive
  omits the referenced license file; this discrepancy remains in the aggregate
  public redistribution review.

The locally prepared wheel is deliberately ignored by Git while the upstream
license discrepancy is reviewed. Maintainers reproduce it by downloading and
checking the pinned source above, inspecting it, and running the stated wheel
build with setuptools 81.0.0 in an isolated build environment. Place the exact
result in this directory before dependency staging. The dependency stager
verifies its hash before binary-only resolution. The private candidate's final
wheelhouse contains it and records this and every resolved wheel hash.
