# macOS App Bundle Signer & Reorganizer

A bash script to clean up, prepare, and codesign macOS application bundles. 

This script is specifically designed to handle common hurdles when signing apps built with cross-platform frameworks (like PyInstaller, Nuitka, Qt, or Electron). It automatically reorganizes misplaced resource files, creates necessary symlinks, recursively signs all embedded binaries (`.dylib`, `.so`), and strictly verifies the final signature.

##  Features
* **PyInstaller/Nuitka Cleanup:** Automatically moves misplaced resources (like `.ico`, `.png`, `.npz`, `.pem`) from the `MacOS` execution directory to the `Resources` directory and leaves valid relative symlinks behind.
* **Recursive Signing:** Finds and safely signs all embedded dynamic libraries and shared objects before signing the main bundle.
* **Interactive Mode:** Automatically queries your local macOS Keychain and provides an interactive menu to select your `Developer ID Application` certificate.
* **CI/CD Ready:** Fully automatable via command-line arguments for seamless integration into GitHub Actions or other build pipelines.
* **Strict Safety:** Built with `set -euo pipefail` to ensure the process immediately aborts if any step fails, preventing the deployment of broken bundles.

---

##  Prerequisites

1. **macOS Environment:** This script must be run on a macOS system (local or CI runner).
2. **Xcode Command Line Tools:** Ensure `codesign` and `security` are available (`xcode-select --install`).
3. **Apple Developer Certificate:** A valid `Developer ID Application` certificate must be installed in your keychain.
4. **Intermediate Certificates:** Ensure the Apple Worldwide Developer Relations (WWDR) G3 and Developer ID G2 intermediate certificates are trusted on your system.

---

##  Installation

Download or copy the `appbundle-signer.sh` script to your repository, typically in a `scripts/` or `packaging/` directory.

Make the script executable:
```bash
chmod +x appbundle-signer.sh
```

---

##  Usage

### 1. Interactive Mode (Local Development)
If you just want to sign an app locally and don't want to copy-paste your long certificate ID, simply provide the path to the app bundle. The script will scan your keychain and let you choose the correct certificate interactively.

```bash
./appbundle-signer.sh -a /path/to/your/SessionPrep.app
```

### 2. Automated Mode (Headless / CI)
For automated builds, provide the exact name or the cryptographic hash of your certificate using the `-c` flag. This bypasses the interactive prompt.

```bash
./appbundle-signer.sh -a "SessionPrep.app" -c "Developer ID Application: Your Name (TEAM_ID)"
```
*Tip: Using the 40-character SHA-1 hash of your certificate (e.g., `A1B2C3D4E5...`) instead of the name is often safer in CI environments to avoid parsing errors with special characters.*

---

##  Integration into GitHub Actions

To use this script in a GitHub Actions workflow, you need to import your signing certificate into a temporary keychain on the macOS runner before executing the script. 

### Step 1: Prepare your Secrets
Export your Apple Certificate as a `.p12` file from your Mac's Keychain Access. Add the following repository secrets in your GitHub repository settings:
* `BUILD_CERTIFICATE_BASE64`: The base64-encoded string of your `.p12` file. *(Generate via terminal: `base64 -i your_cert.p12 | pbcopy`)*
* `P12_PASSWORD`: The password you set when exporting the `.p12` file.
* `DEVELOPER_ID_HASH`: The 40-character SHA-1 hash of your certificate, or the exact name string.

### Step 2: The Workflow YAML
Here is a complete step-by-step job for your `.github/workflows/build.yml`:

```yaml
name: Build and Sign macOS App

on:
  push:
    branches: [ "main" ]
  release:
    types: [ "created" ]

jobs:
  build-and-sign:
    runs-on: macos-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # (Insert your build steps here, e.g., PyInstaller, py2app, etc.)
      # - name: Build App
      #   run: pyinstaller your_script.spec

      - name: Install Apple Certificate
        env:
          BUILD_CERTIFICATE_BASE64: ${{ secrets.BUILD_CERTIFICATE_BASE64 }}
          P12_PASSWORD: ${{ secrets.P12_PASSWORD }}
          KEYCHAIN_PASSWORD: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Create variables
          CERTIFICATE_PATH=$RUNNER_TEMP/build_certificate.p12
          KEYCHAIN_PATH=$RUNNER_TEMP/app-signing.keychain-db

          # Decode base64 certificate
          echo -n "$BUILD_CERTIFICATE_BASE64" | base64 --decode -o $CERTIFICATE_PATH

          # Create temporary keychain
          security create-keychain -p "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH
          security set-keychain-settings -lut 21600 $KEYCHAIN_PATH
          security unlock-keychain -p "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH

          # Import certificate to keychain
          security import $CERTIFICATE_PATH -P "$P12_PASSWORD" -A -t cert -f pkcs12 -k $KEYCHAIN_PATH
          security list-keychain -d user -s $KEYCHAIN_PATH

          # Allow codesign to access the keychain without a UI prompt
          security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KEYCHAIN_PASSWORD" $KEYCHAIN_PATH

      - name: Reorganize and Sign App Bundle
        env:
          CERT_HASH: ${{ secrets.DEVELOPER_ID_HASH }}
        run: |
          chmod +x ./scripts/appbundle-signer.sh
          ./scripts/appbundle-signer.sh -a "dist/SessionPrep.app" -c "$CERT_HASH"

      # (Optional: Add your notarization step using xcrun notarytool here)
```

---

##  Troubleshooting

* **`errSecInternalComponent` / `unable to build chain to self-signed root`:** Your macOS system is missing the necessary intermediate certificates to verify the Developer ID. Download the **Developer ID - G2** and **Apple Root CA - G2** certificates from the [Apple PKI portal](https://www.apple.com/certificateauthority/) and install them in your keychain.
  
* **`code object is not signed at all` in the MacOS folder:**
  Apple strictly enforces that the `Contents/MacOS` directory contains *only* executable binaries. If you add new data files (like `.json`, `.png`, or `.xml`) to your app via PyInstaller, you must update the `reorganize_resources` function in the script to move those files to the `Contents/Resources` directory and symlink them back.

