#!/usr/bin/env bash
# Publish GitHub Actions build artifacts to a GitHub draft release.
#
# The script is intentionally usable outside GitHub Actions. In --dry-run mode
# it validates inputs, filters artifacts, and stages selected files without
# requiring GitHub authentication.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_ALLOWLIST="$SCRIPT_DIR/release-assets.allowlist"

ARTIFACT_ROOT=""
DOWNLOAD_RUN_ID=""
ARTIFACT_PATTERN="*"
ALLOWLIST="$DEFAULT_ALLOWLIST"
STAGING_DIR="release-assets"
MODE=""
REF_NAME=""
TARGET_SHA=""
REPO=""
RELEASE_TAG=""
RELEASE_TITLE=""
DRY_RUN=0

PATTERNS=()
ASSET_PATHS=()
ASSET_NAMES=()

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  publish-draft-release.sh [--artifact-root DIR] --mode branch|tag --ref-name NAME \
    --target SHA --repo OWNER/REPO [options]

Required:
  --mode branch|tag        Release replacement semantics.
  --ref-name NAME          GitHub ref name. Branch name in branch mode,
                           existing Git tag name in tag mode.
  --target SHA             Commit SHA this build/release represents. In branch
                           mode the synthetic release tag is created at this
                           commit. In tag mode it is logged for traceability;
                           the existing Git tag selected by --ref-name is used
                           and verified by GitHub before publishing.
  --repo OWNER/REPO        GitHub repository to publish into.

Artifact source:
  --artifact-root DIR      Directory containing downloaded artifacts, or the
                           destination when --download-run-id is used.
                           Default with --download-run-id: downloaded-artifacts
  --download-run-id ID     Download artifacts from this GitHub Actions run first.
  --artifact-pattern GLOB  Artifact name pattern used for downloads.
                           Default: *

Options:
  --allowlist FILE         Glob allowlist. Default: release-assets.allowlist
                           next to this script.
  --staging-dir DIR        Clean output directory for selected release assets. Default: release-assets
  --release-tag TAG        Override generated release tag.
  --release-title TITLE    Override generated release title.
  --dry-run                Select and stage files, but do not call GitHub.
  -h, --help               Show this help.

Branch mode:
  Replaces a synthetic draft release/tag derived from the branch name.

Tag mode:
  Replaces only an existing draft release for the real tag, or creates one.
  Published releases are never modified.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact-root) ARTIFACT_ROOT="${2:-}"; shift 2 ;;
        --download-run-id) DOWNLOAD_RUN_ID="${2:-}"; shift 2 ;;
        --artifact-pattern) ARTIFACT_PATTERN="${2:-}"; shift 2 ;;
        --allowlist) ALLOWLIST="${2:-}"; shift 2 ;;
        --staging-dir) STAGING_DIR="${2:-}"; shift 2 ;;
        --mode) MODE="${2:-}"; shift 2 ;;
        --ref-name) REF_NAME="${2:-}"; shift 2 ;;
        --target) TARGET_SHA="${2:-}"; shift 2 ;;
        --repo) REPO="${2:-}"; shift 2 ;;
        --release-tag) RELEASE_TAG="${2:-}"; shift 2 ;;
        --release-title) RELEASE_TITLE="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

validate_basic_inputs() {
    if [[ -z "$ARTIFACT_ROOT" && -n "$DOWNLOAD_RUN_ID" ]]; then
        ARTIFACT_ROOT="downloaded-artifacts"
    fi

    [[ -n "$ARTIFACT_ROOT" ]] || die "--artifact-root is required unless --download-run-id is used"
    [[ -n "$MODE" ]] || die "--mode is required"
    [[ -n "$REF_NAME" ]] || die "--ref-name is required"
    [[ -n "$TARGET_SHA" ]] || die "--target is required"
    [[ -n "$REPO" ]] || die "--repo is required"
    [[ -n "$ARTIFACT_PATTERN" ]] || die "--artifact-pattern must not be empty"

    [[ "$MODE" == "branch" || "$MODE" == "tag" ]] || die "--mode must be 'branch' or 'tag'"
    [[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "--repo must look like OWNER/REPO"
    [[ "$TARGET_SHA" =~ ^[0-9A-Fa-f]{7,40}$ ]] || die "--target must be a Git commit SHA"
    [[ -z "$DOWNLOAD_RUN_ID" || "$DOWNLOAD_RUN_ID" =~ ^[0-9]+$ ]] || die "--download-run-id must be numeric"
    [[ "$REF_NAME" != -* ]] || die "--ref-name must not start with '-'"
    [[ "$REF_NAME" != *".."* ]] || die "--ref-name must not contain '..'"
    [[ "$REF_NAME" != *$'\n'* && "$REF_NAME" != *$'\r'* ]] || die "--ref-name must be a single line"
    [[ "$ARTIFACT_ROOT" != *$'\n'* && "$ARTIFACT_ROOT" != *$'\r'* ]] || die "--artifact-root must be a single line"
    [[ "$ARTIFACT_PATTERN" != *$'\n'* && "$ARTIFACT_PATTERN" != *$'\r'* ]] || die "--artifact-pattern must be a single line"

    [[ -f "$ALLOWLIST" ]] || die "Allowlist does not exist: $ALLOWLIST"
    [[ -n "$STAGING_DIR" && "$STAGING_DIR" != "/" ]] || die "Unsafe --staging-dir: $STAGING_DIR"
    [[ "$STAGING_DIR" != *$'\n'* && "$STAGING_DIR" != *$'\r'* ]] || die "--staging-dir must be a single line"
}

canonical_target_path() {
    local path="$1"
    local label="$2"
    local parent
    local base

    parent="$(dirname "$path")"
    base="$(basename "$path")"
    [[ -d "$parent" ]] || die "Parent directory for $label does not exist: $parent"

    parent="$(cd "$parent" && pwd -P)"
    printf '%s/%s' "$parent" "$base"
}

validate_artifact_root_for_download() {
    local artifact_root_real
    local cwd_real

    [[ -n "$ARTIFACT_ROOT" && "$ARTIFACT_ROOT" != "/" ]] || die "Unsafe --artifact-root: $ARTIFACT_ROOT"

    artifact_root_real="$(canonical_target_path "$ARTIFACT_ROOT" "--artifact-root")"
    cwd_real="$(pwd -P)"

    [[ "$artifact_root_real" != "$cwd_real" ]] || die "--artifact-root must not be the working directory"
    case "$cwd_real/" in
        "$artifact_root_real"/*)
            die "--artifact-root must not contain the working directory"
            ;;
    esac
}

validate_existing_artifact_root() {
    [[ -d "$ARTIFACT_ROOT" ]] || die "Artifact root does not exist or is not a directory: $ARTIFACT_ROOT"
}

validate_staging_dir() {
    local artifact_root_real
    local staging_real

    artifact_root_real="$(cd "$ARTIFACT_ROOT" && pwd -P)"
    staging_real="$(canonical_target_path "$STAGING_DIR" "--staging-dir")"

    [[ "$staging_real" != "$artifact_root_real" ]] || die "--staging-dir must not be the artifact root"
    case "$staging_real" in
        "$artifact_root_real"/*)
            die "--staging-dir must not be inside --artifact-root"
            ;;
    esac
}

validate_gh_environment() {
    require_command gh
    gh auth status >/dev/null 2>&1 || die "GitHub CLI is not authenticated. Set GH_TOKEN in CI or run 'gh auth login' locally."
}

download_artifacts() {
    [[ -n "$DOWNLOAD_RUN_ID" ]] || return 0

    validate_gh_environment
    validate_artifact_root_for_download

    log "Downloading artifacts from run $DOWNLOAD_RUN_ID using pattern '$ARTIFACT_PATTERN'"
    rm -rf -- "$ARTIFACT_ROOT"
    mkdir -p "$ARTIFACT_ROOT"
    gh run download "$DOWNLOAD_RUN_ID" \
        --repo "$REPO" \
        --pattern "$ARTIFACT_PATTERN" \
        --dir "$ARTIFACT_ROOT"
}

sanitize_ref_name() {
    local value="$1"
    value="$(printf '%s' "$value" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
    value="${value:0:80}"
    [[ -n "$value" ]] || value="unnamed"
    printf '%s' "$value"
}

derive_release_metadata() {
    if [[ -z "$RELEASE_TAG" ]]; then
        if [[ "$MODE" == "tag" ]]; then
            RELEASE_TAG="$REF_NAME"
        else
            RELEASE_TAG="branch-build-$(sanitize_ref_name "$REF_NAME")"
        fi
    fi

    [[ "$RELEASE_TAG" != -* ]] || die "Release tag must not start with '-'"
    [[ "$RELEASE_TAG" != *".."* ]] || die "Release tag must not contain '..'"
    [[ "$RELEASE_TAG" != *$'\n'* && "$RELEASE_TAG" != *$'\r'* ]] || die "Release tag must be a single line"

    if [[ -z "$RELEASE_TITLE" ]]; then
        if [[ "$MODE" == "tag" ]]; then
            RELEASE_TITLE="$RELEASE_TAG"
        else
            RELEASE_TITLE="Branch build: $REF_NAME"
        fi
    fi
}

load_allowlist() {
    local line
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
        [[ -n "$line" ]] || continue
        PATTERNS+=("$line")
    done < "$ALLOWLIST"

    [[ ${#PATTERNS[@]} -gt 0 ]] || die "Allowlist contains no patterns: $ALLOWLIST"
}

matches_allowlist() {
    local name="$1"
    local pattern
    for pattern in "${PATTERNS[@]}"; do
        if [[ "$name" == $pattern ]]; then
            return 0
        fi
    done
    return 1
}

asset_name_seen() {
    local name="$1"
    local existing
    set +u
    for existing in "${ASSET_NAMES[@]}"; do
        if [[ "$existing" == "$name" ]]; then
            set -u
            return 0
        fi
    done
    set -u
    return 1
}

stage_asset_stream() {
    local name="$1"
    local source_label="$2"
    local output="$STAGING_DIR/$name"

    matches_allowlist "$name" || return 0
    asset_name_seen "$name" && die "Duplicate release asset filename: $name (from $source_label)"

    mkdir -p "$STAGING_DIR"
    cat > "$output"
    [[ -s "$output" ]] || die "Selected asset is empty after staging: $source_label"

    ASSET_NAMES+=("$name")
    ASSET_PATHS+=("$output")
    log "Selected $name from $source_label"
}

scan_regular_file() {
    local file="$1"
    local name
    name="$(basename "$file")"
    matches_allowlist "$name" || return 0
    stage_asset_stream "$name" "$file" < "$file"
}

scan_zip_file() {
    local zip_file="$1"
    local entry
    local list_file
    log "Inspecting ZIP artifact: $zip_file"

    list_file="$(mktemp "${TMPDIR:-/tmp}/github-release-zip-list.XXXXXX")"
    if ! unzip -Z1 "$zip_file" > "$list_file" 2>/dev/null; then
        rm -f "$list_file"
        die "Could not read ZIP artifact: $zip_file"
    fi

    while IFS= read -r entry || [[ -n "$entry" ]]; do
        [[ -n "$entry" ]] || continue
        [[ "$entry" != */ ]] || continue
        local name
        name="$(basename "$entry")"
        matches_allowlist "$name" || continue
        local tmp_entry
        tmp_entry="$(mktemp "${TMPDIR:-/tmp}/github-release-asset.XXXXXX")"
        if ! unzip -p "$zip_file" "$entry" > "$tmp_entry"; then
            rm -f "$tmp_entry"
            die "Could not extract '$entry' from ZIP artifact: $zip_file"
        fi
        stage_asset_stream "$name" "$zip_file:$entry" < "$tmp_entry"
        rm -f "$tmp_entry"
    done < "$list_file"
    rm -f "$list_file"
}

select_assets() {
    rm -rf -- "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"

    local file
    while IFS= read -r -d '' file; do
        case "$file" in
            "$STAGING_DIR"/*) continue ;;
        esac
        if [[ "$file" == *.zip ]]; then
            scan_zip_file "$file"
        else
            scan_regular_file "$file"
        fi
    done < <(find "$ARTIFACT_ROOT" -type f -print0)

    [[ ${#ASSET_PATHS[@]} -gt 0 ]] || die "No release assets matched allowlist '$ALLOWLIST' under '$ARTIFACT_ROOT'"

    log "Staged ${#ASSET_PATHS[@]} release asset(s) in $STAGING_DIR:"
    local asset
    for asset in "${ASSET_PATHS[@]}"; do
        printf '  %s\n' "$asset"
    done
}

validate_publish_environment() {
    validate_gh_environment
}

release_json_field() {
    local tag="$1"
    local jq_expr="$2"
    gh api "repos/$REPO/releases/tags/$tag" --jq "$jq_expr" 2>/dev/null || return 1
}

release_exists() {
    release_json_field "$1" '.id' >/dev/null
}

release_is_draft() {
    [[ "$(release_json_field "$1" '.draft')" == "true" ]]
}

delete_release_if_exists() {
    local tag="$1"
    local release_id
    release_id="$(release_json_field "$tag" '.id')" || return 0
    gh api -X DELETE "repos/$REPO/releases/$release_id" --silent
}

delete_tag_ref_if_exists() {
    local tag="$1"
    gh api -X DELETE "repos/$REPO/git/refs/tags/$tag" --silent >/dev/null 2>&1 || true
}

create_draft_release() {
    local notes
    notes="Automated draft release generated from $REF_NAME at $TARGET_SHA."

    if [[ "$MODE" == "tag" ]]; then
        gh release create "$RELEASE_TAG" \
            --repo "$REPO" \
            --draft \
            --verify-tag \
            --title "$RELEASE_TITLE" \
            --notes "$notes"
    else
        gh release create "$RELEASE_TAG" \
            --repo "$REPO" \
            --draft \
            --target "$TARGET_SHA" \
            --title "$RELEASE_TITLE" \
            --notes "$notes"
    fi
}

publish_release() {
    validate_publish_environment

    if [[ "$MODE" == "branch" ]]; then
        if release_exists "$RELEASE_TAG"; then
            release_is_draft "$RELEASE_TAG" || die "Refusing to replace published branch release: $RELEASE_TAG"
            log "Deleting existing draft branch release: $RELEASE_TAG"
            delete_release_if_exists "$RELEASE_TAG"
        fi
        log "Deleting existing synthetic branch tag if present: $RELEASE_TAG"
        delete_tag_ref_if_exists "$RELEASE_TAG"
    else
        if release_exists "$RELEASE_TAG"; then
            release_is_draft "$RELEASE_TAG" || die "Refusing to modify published tag release: $RELEASE_TAG"
            log "Deleting existing draft tag release while preserving real tag: $RELEASE_TAG"
            delete_release_if_exists "$RELEASE_TAG"
        fi
    fi

    log "Creating draft release: $RELEASE_TAG"
    create_draft_release

    log "Uploading ${#ASSET_PATHS[@]} asset(s)"
    gh release upload "$RELEASE_TAG" "${ASSET_PATHS[@]}" --repo "$REPO" --clobber
    log "Draft release ready: $RELEASE_TAG"
}

main() {
    require_command sed
    require_command find
    require_command basename
    require_command dirname
    require_command unzip

    validate_basic_inputs
    download_artifacts
    validate_existing_artifact_root
    validate_staging_dir
    derive_release_metadata
    load_allowlist
    select_assets

    log "Release mode: $MODE"
    log "Release tag: $RELEASE_TAG"
    log "Release title: $RELEASE_TITLE"
    log "Repository: $REPO"
    log "Target: $TARGET_SHA"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry run complete. No GitHub release was changed."
        return 0
    fi

    publish_release
}

main "$@"
