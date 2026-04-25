# Release Strategy

Recommended distribution model:

1. keep runtime code in this repository
2. publish production model bundles as GitHub Releases or Hugging Face artifacts
3. make the CLI resolve and download a chosen bundle version
4. later add Homebrew and curl-based installers

## Current Constraint

The current production zip is too large for a comfortable GitHub Release asset workflow. The next practical move is to publish bundle metadata in GitHub and host the heavy artifact on Hugging Face or equivalent object storage.
