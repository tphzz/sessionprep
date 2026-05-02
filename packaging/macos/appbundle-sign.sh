#!/bin/bash

# ==============================================================================
# Strict Mode: Aborts the script immediately if an error occurs
# ==============================================================================
set -euo pipefail

# ==============================================================================
# Color definitions for clear logging
# ==============================================================================
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
# Help function / Usage
# ==============================================================================
usage() {
    echo -e "Usage: $0 -a <Path_to_App> [-c <Certificate_ID_or_Hash>]"
    echo -e "Example (Interactive):"
    echo -e "  $0 -a SessionPrep.app"
    echo -e "Example (Automated/CI):"
    echo -e "  $0 -a SessionPrep.app -c \"Developer ID Application: Benjamin Zeiss (Y6XM7FZX63)\"\n"
    exit 1
}

# ==============================================================================
# Parse parameters
# ==============================================================================
APP_PATH=""
CERT_NAME=""

while getopts "a:c:h" opt; do
    case $opt in
        a) APP_PATH="$OPTARG" ;;
        c) CERT_NAME="$OPTARG" ;;
        h) usage ;;
        \?) usage ;;
    esac
done

# Check if required app path is provided
if [ -z "$APP_PATH" ]; then
    log_error "Missing parameter. App path (-a) is required."
fi

# ==============================================================================
# Path validation
# ==============================================================================
if [ ! -d "$APP_PATH" ]; then
    log_error "App bundle not found at: $APP_PATH"
fi

# ==============================================================================
# FUNCTION: Select Certificate Interactively
# ==============================================================================
select_certificate() {
    log_info "No certificate provided via -c. Querying local keychain..."
    
    # Query keychain and filter for Developer ID. 
    # '|| true' prevents the strict mode from killing the script if grep finds nothing.
    local cert_raw
    cert_raw=$(security find-identity -v -p codesigning | grep "Developer ID Application" || true)

    if [ -z "$cert_raw" ]; then
        log_error "No 'Developer ID Application' certificates found in your keychain."
    fi

    # Read output into arrays
    local cert_hashes=()
    local cert_names=()
    
    while IFS= read -r line; do
        # Extract the hex hash (2nd column in security output)
        local hash=$(echo "$line" | awk '{print $2}')
        # Extract the human-readable name inside the quotes
        local name=$(echo "$line" | grep -o '".*"' | sed 's/"//g')
        
        cert_hashes+=("$hash")
        cert_names+=("$name")
    done <<< "$cert_raw"

    # Auto-select if there is exactly one certificate
    if [ ${#cert_hashes[@]} -eq 1 ]; then
        CERT_NAME="${cert_hashes[0]}"
        log_success "Auto-selected the only available certificate: ${cert_names[0]}"
        return
    fi

    # Interactive menu if multiple certificates are found
    echo -e "${YELLOW}Multiple Developer ID certificates found. Please choose one:${NC}"
    
    # Customize the prompt text for the 'select' menu
    PS3="Enter the number of the certificate to use (1-${#cert_names[@]}): "
    
    select opt in "${cert_names[@]}"; do
        if [ -n "$opt" ]; then
            # Find the index of the selected option
            for i in "${!cert_names[@]}"; do
                if [ "${cert_names[$i]}" = "$opt" ]; then
                    # We use the HEX HASH for codesign. It is much safer than the string!
                    CERT_NAME="${cert_hashes[$i]}"
                    log_success "Selected: $opt"
                    break 2
                fi
            done
        else
            log_warn "Invalid selection. Please try again."
        fi
    done
}

# ==============================================================================
# FUNCTION: Reorganize Resources (PyInstaller Cleanup)
# ==============================================================================
reorganize_resources() {
    local app="$1"
    local macos_dir="$app/Contents/MacOS"
    local resources_dir="$app/Contents/Resources"

    log_info "Starting app bundle cleanup for: $app"

    # 1. Move sessionprepgui/res
    if [ -d "$macos_dir/sessionprepgui/res" ] && [ ! -L "$macos_dir/sessionprepgui/res" ]; then
        mkdir -p "$resources_dir/sessionprepgui"
        mv "$macos_dir/sessionprepgui/res" "$resources_dir/sessionprepgui/"
        ln -s "../../Resources/sessionprepgui/res" "$macos_dir/sessionprepgui/res"
        log_success "res directory successfully moved and linked."
    else
        log_info "res directory already linked or not found."
    fi

    # 2. Move grpc roots.pem
    if [ -f "$macos_dir/grpc/_cython/_credentials/roots.pem" ] && [ ! -L "$macos_dir/grpc/_cython/_credentials/roots.pem" ]; then
        mkdir -p "$resources_dir/grpc/_cython/_credentials"
        mv "$macos_dir/grpc/_cython/_credentials/roots.pem" "$resources_dir/grpc/_cython/_credentials/"
        ln -s "../../../../Resources/grpc/_cython/_credentials/roots.pem" "$macos_dir/grpc/_cython/_credentials/roots.pem"
        log_success "roots.pem successfully moved and linked."
    else
        log_info "roots.pem already linked or not found."
    fi

    # 3. Move SciPy sobol_direction_numbers.npz
    if [ -f "$macos_dir/scipy/stats/_sobol_direction_numbers.npz" ] && [ ! -L "$macos_dir/scipy/stats/_sobol_direction_numbers.npz" ]; then
        mkdir -p "$resources_dir/scipy/stats"
        mv "$macos_dir/scipy/stats/_sobol_direction_numbers.npz" "$resources_dir/scipy/stats/"
        ln -s "../../../Resources/scipy/stats/_sobol_direction_numbers.npz" "$macos_dir/scipy/stats/_sobol_direction_numbers.npz"
        log_success "SciPy npz file successfully moved and linked."
    else
        log_info "SciPy npz file already linked or not found."
    fi
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Step 0: Resolve Certificate
if [ -z "$CERT_NAME" ]; then
    select_certificate
fi

# Step 1: Execute the built-in reorganizer function
reorganize_resources "$APP_PATH"

# Step 2: Recursively sign all .dylib and .so files
log_info "Finding and signing .dylib and .so files in $APP_PATH..."
while IFS= read -r -d '' file; do
    codesign --force --verify --timestamp --options runtime --sign "$CERT_NAME" "$file"
done < <(find "$APP_PATH" -type f \( -name "*.dylib" -o -name "*.so" \) -print0)
log_success "All embedded libraries were successfully signed."

# Step 3: Sign the main app bundle
log_info "Signing the main app bundle: $APP_PATH"
codesign --force --verify --verbose --timestamp --options runtime --sign "$CERT_NAME" "$APP_PATH"

# Step 4: Final verification
log_info "Performing final strict signature verification..."
codesign -vvv --deep --strict "$APP_PATH"

log_success "App bundle successfully signed and verified!"

