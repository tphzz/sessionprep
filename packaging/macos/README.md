# macOS App Deployment Suite: Sign, Notarize & Package

A toolset for preparing, signing, and distributing macOS application bundles. This suite is specifically designed to handle the complexities of cross-platform frameworks (like **PyInstaller**, **Nuitka**, or **PySide6/Qt**) while automating the Apple Notarization and DMG creation process.

## The Deployment Suite

The suite consists of two specialized scripts:

1.  **appbundle-sign.sh**: The "workhorse" for development. It reorganizes the bundle structure, moves misplaced resources, manages symlinks, and performs recursive signing of all embedded binaries.
2.  **appbundle-release.sh**: The "finisher" for production. It utilizes the signing script, communicates with the Apple Notary Service, staples the notarization ticket, and packages everything into a user-friendly DMG.

---

## Features

* **Structural Repair:** Automatically moves non-executable resources (like `.ico`, `.png`, `.pem`, `.npz`) from `MacOS/` to `Resources/` and maintains functionality via relative symlinks.
* **Hardened Runtime Support:** Applies necessary entitlements to comply with modern macOS security requirements.
* **Modern Notarization:** Uses Apple's `notarytool` (Xcode 13+) for faster, reliable automated security scans.
* **Professional DMG Packaging:** Creates a signed and notarized Disk Image including a shortcut to the `/Applications` folder for a standard drag-and-drop installation experience.
* **CI/CD Maturity:** Uses scoped environment variables to prevent naming conflicts in complex build pipelines.
* **Pre-flight Validation:** Performs a dashboard-style check of all credentials and system tools before starting long-running tasks.

---

## Configuration (Environment Variables)

To enable headless/automated operation, the suite looks for the following scoped variables. It is recommended to export these in your `.zshrc` (locally) or set them as **GitHub Repository Secrets**.

### For Signing:
* **`MACOS_DEV_ID_CERT_NAME`**: The exact name of your certificate (e.g., `Developer ID Application: Your Name (TEAM_ID)`) or its 40-character SHA-1 hash.

### For Notarization & Distribution:
* **`APPLE_NOTARY_USER`**: Your Apple ID email address.
* **`APPLE_NOTARY_APP_PASSWORD`**: An **App-Specific Password** generated at [appleid.apple.com](https://appleid.apple.com). Do **not** use your primary iCloud password.
* **`APPLE_TEAM_ID`**: Your 10-character Team ID found in the Apple Developer Portal.

---

## Usage

### 1. Development: Structural Signing
Use this during active development to verify that the app bundle structure is correct and that the app starts without "Team ID mismatch" or "translocation" errors.

```bash
chmod +x packaging/macos/appbundle-sign.sh
./packaging/macos/appbundle-sign.sh -a "dist/SessionPrep.app"
```

### 2. Production: The Full Release
Use this when you are ready to distribute your software. This script handles the notarization wait-time and creates the final DMG.

```bash
chmod +x packaging/macos/appbundle-release.sh
./packaging/macos/appbundle-release.sh -a "dist/SessionPrep.app"
```

---

## Integration into GitHub Actions

The suite is designed to be the final stage of a CI/CD pipeline.

### Step 1: Secrets Setup
Add the following to your GitHub Repository Secrets:
* `APPLE_NOTARY_USER`
* `APPLE_NOTARY_APP_PASSWORD`
* `APPLE_TEAM_ID`
* `MACOS_DEV_ID_CERT_NAME`
* `P12_BASE64` (Your `.p12` certificate file encoded as base64)
* `P12_PASSWORD` (The password for your `.p12` file)

### Step 2: Workflow Logic
1.  **Build:** Create the `.app` bundle (e.g., via PyInstaller).
2.  **Certificate Import:** Use a GitHub Action (like `apple-actions/import-codesign-certs`) to load your certificate into the runner's keychain.
3.  **Signing:** Run `appbundle-sign.sh` for all builds to verify structural integrity.
4.  **Release:** On **Tags** or **Main** branch merges, run `appbundle-release.sh` to generate the notarized DMG.

---

## Troubleshooting

> **Pre-flight Check Fails:** > Ensure your variables are exported in your shell. Use `export APPLE_NOTARY_USER="..."` instead of just assigning the value.

> **Notarization Rejected:** > If Apple rejects the submission, check the logs. Common reasons include unsigned third-party libraries or resources still residing in the `Contents/MacOS` folder. The `appbundle-sign.sh` script is designed to prevent these issues.

> **DMG Assessment Failed:** > If the final `spctl` check fails, ensure your certificate is a **Developer ID Application** type. Standard "Apple Development" or "Mac App Store" certificates are not valid for notarized direct distribution.

