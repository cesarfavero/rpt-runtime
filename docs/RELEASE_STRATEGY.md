# Release Strategy

Recommended distribution model:

1. keep runtime code in this repository
2. publish production model bundles as GitHub Releases or Hugging Face artifacts
3. make the CLI resolve and download a chosen bundle version
4. later add Homebrew and curl-based installers
