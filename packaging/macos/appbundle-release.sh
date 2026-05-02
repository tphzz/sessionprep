#!/bin/bash
# ==============================================================================
# macOS App Release Script (Notarization + DMG Creation)
# Supports: Local Interactive Use & CI/CD (GitHub Actions)
# ==============================================================================
set -euo pipefail

# Color definitions for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ==============================================================================
# Configuration & Inputs
# ==============================================================================
APP_PATH=""
# Scoped variables to avoid overlap with App Store / iOS distribution
CERT_NAME="${MACOS_DEV_ID_CERT_NAME:-}"
APPLE_ID="${APPLE_NOTARY_USER:-}"
APPLE_PASSWORD="${APPLE_NOTARY_APP_PASSWORD:-}"
TEAM_ID="${APPLE_TEAM_ID:-Y6XM7FZX63}"

usage() {
    echo -e "Usage: $0 -a <App_Path> [-c <Cert_Name>] [-u <Apple_ID>] [-p <App_Password>] [-t <Team_ID>]"
    echo -e "\nNote: Environment variables are preferred for CI/CD:"
    echo -e "  MACOS_DEV_ID_CERT_NAME, APPLE_NOTARY_USER, APPLE_NOTARY_APP_PASSWORD, APPLE_TEAM_ID"
    exit 1
}

while getopts "a:c:u:p:t:h" opt; do
    case $opt in
        a) APP_PATH="$OPTARG" ;;
        c) CERT_NAME="$OPTARG" ;;
        u) APPLE_ID="$OPTARG" ;;
        p) APPLE_PASSWORD="$OPTARG" ;;
        t) TEAM_ID="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ==============================================================================
# Maturity Check: Pre-flight Validation
# ==============================================================================
validate_environment() {
    local missing=0
    echo -e "${BLUE}=== Pre-flight Environment Validation ===${NC}"
    
    # Internal helper to check and mask sensitive data
    check_status() {
        local label=$1
        local value=$2
        local is_password=$3
        
        if [ -z "$value" ]; then
            echo -e "  [${RED}MISSING${NC}] $label"
            missing=1
        else
            if [ "$is_password" -eq 1 ]; then
                echo -e "  [${GREEN}SET${NC}]     $label: ****************"
            else
                echo -e "  [${GREEN}SET${NC}]     $label: $value"
            fi
        fi
    }

    check_status "App Path (-a)" "$APP_PATH" 0
    check_status "Certificate (MACOS_DEV_ID_CERT_NAME)" "$CERT_NAME" 0
    check_status "Apple ID (APPLE_NOTARY_USER)" "$APPLE_ID" 0
    check_status "App Password (APPLE_NOTARY_APP_PASSWORD)" "$APPLE_PASSWORD" 1
    check_status "Team ID (APPLE_TEAM_ID)" "$TEAM_ID" 0

    # Check for required system tools
    echo -e "\n${BLUE}=== System Tools Check ===${NC}"
    for tool in xcrun ditto hdiutil codesign; do
        if command -v "$tool" >/dev/null 2>&1; then
            echo -e "  [${GREEN}OK${NC}]      $tool"
        else
            echo -e "  [${RED}MISSING${NC}] $tool"
            missing=1
        fi
    done

    if [ "$missing" -eq 1 ]; then
        echo ""
        log_error "Pre-flight check failed. Please provide all required variables/tools."
    fi
    log_success "All requirements met. Proceeding to release...\n"
}

# Run the validation
validate_environment

# Set derived paths
BUNDLE_NAME=$(basename "$APP_PATH" .app)
DMG_PATH="${BUNDLE_NAME}.dmg"

# ==============================================================================
# Step 1: Notarize the .app (via Zip submission)
# ==============================================================================
ZIP_PATH="${BUNDLE_NAME}_submit.zip"
log_info "Creating temporary zip for .app notarization..."
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

log_info "Submitting .app to Apple Notary Service..."
xcrun notarytool submit "$ZIP_PATH" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_PASSWORD" \
    --team-id "$TEAM_ID" \
    --wait

log_success ".app notarization approved."
log_info "Stapling .app bundle..."
xcrun stapler staple "$APP_PATH"
rm "$ZIP_PATH"

# ==============================================================================
# Step 2: Create the DMG
# ==============================================================================
log_info "Creating DMG: $DMG_PATH..."
if [ -f "$DMG_PATH" ]; then rm "$DMG_PATH"; fi

# Simple hdiutil creation. UDZO format is compressed and read-only.
hdiutil create -volname "$BUNDLE_NAME" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"

# ==============================================================================
# Step 3: Sign the DMG
# ==============================================================================
log_info "Signing the DMG file..."
codesign --force --verify --timestamp --sign "$CERT_NAME" "$DMG_PATH"

# ==============================================================================
# Step 4: Notarize the DMG
# ==============================================================================
log_info "Submitting DMG to Apple Notary Service..."
xcrun notarytool submit "$DMG_PATH" \
    --apple-id "$APPLE_ID" \
    --password "$APPLE_PASSWORD" \
    --team-id "$TEAM_ID" \
    --wait

log_success "DMG notarization approved."
log_info "Stapling the DMG..."
xcrun stapler staple "$DMG_PATH"

# ==============================================================================
# Final Verification
# ==============================================================================
log_info "Performing final Gatekeeper assessment on DMG..."
spctl --assess -vv --type install "$DMG_PATH"

log_success "RELEASE COMPLETE: $DMG_PATH is ready for distribution."

