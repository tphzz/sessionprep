"""Pure ownership and ordering rules for unpublished GitHub release replacement."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import os
import subprocess
import time
import uuid
from urllib.parse import quote, urlsplit


_UPLOAD_ATTEMPTS = 3
_TRANSIENT_UPLOAD_STATUS = {"408", "500", "502", "503", "504"}
_TRANSIENT_CURL_ERRORS = {5, 6, 7, 18, 28, 52, 55, 56, 92}


def branch_tag(ref_name):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ref_name).strip("-")[:60] or "unnamed"
    fingerprint = hashlib.sha256(ref_name.encode("utf-8")).hexdigest()[:16]
    return f"branch-build-{slug}-{fingerprint}"


def replacement_plan(releases, mode, ref_name, release_tag, producer_run_id):
    marker = f"<!-- github-draft-release-key: branch:{ref_name} -->"
    selected = []
    newest = 0
    owns_tag = False
    for release in releases:
        body = release.get("body") or ""
        owned = marker in body
        matching_tag = release.get("tag_name") == release_tag
        if not (matching_tag or (mode == "branch" and owned)):
            continue
        if release.get("draft") is not True:
            raise ValueError(f"Refusing to replace published {mode} release: {release.get('id')}")
        if mode == "branch" and not owned:
            raise ValueError("Release tag collision without matching branch ownership")
        identifier = release.get("id")
        if type(identifier) is not int or identifier <= 0:
            raise ValueError("Invalid GitHub release identifier")
        selected.append(identifier)
        owns_tag |= matching_tag and owned
        if mode == "branch":
            match = re.search(r"<!-- sessionprep-run: ([0-9]+) -->", body)
            if "<!-- sessionprep-candidate -->" not in body:
                newest = max(newest, int(match[1]) if match else 0)
    skip = mode == "branch" and newest > producer_run_id
    return {"delete": [] if skip else selected, "skip": skip, "owns_tag": owns_tag}


def assert_draft_unchanged(current, snapshot):
    if (current.get("draft") is not True or current.get("tag_name") != snapshot.get("tag_name")
            or current.get("body") != snapshot.get("body")):
        raise ValueError("Release changed or was published; refusing automatic deletion")


def assert_deletable(current, snapshot, mode, ref_name):
    """Recheck user-visible state immediately before a destructive operation."""
    assert_draft_unchanged(current, snapshot)
    if mode == "branch" and f"<!-- github-draft-release-key: branch:{ref_name} -->" not in (current.get("body") or ""):
        raise ValueError("Draft ownership changed; refusing automatic deletion")


class GitHub:
    def __init__(self, repository):
        self.root = f"repos/{repository}"

    def api(self, path, method="GET", payload=None, pages=False):
        command = ["gh", "api", "--method", method, path]
        if pages:
            command.extend(["--paginate", "--slurp"])
        if payload is not None:
            command.extend(["--input", "-"])
        result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"GitHub {method} failed: {result.stderr.strip()}")
        return json.loads(result.stdout) if result.stdout.strip() else None

    def releases(self):
        pages = self.api(self.root + "/releases?per_page=100", pages=True)
        return [item for page in pages for item in page]

    def get_release(self, identifier):
        return self.api(self.root + f"/releases/{identifier}")

    def create_release(self, payload):
        return self.api(self.root + "/releases", "POST", payload)

    def update_release(self, identifier, payload):
        return self.api(self.root + f"/releases/{identifier}", "PATCH", payload)

    def delete_release(self, identifier):
        self.api(self.root + f"/releases/{identifier}", "DELETE")

    def branch_tag_exists(self, tag):
        return any(ref.get("ref") == "refs/tags/" + tag for ref in self.api(
            self.root + "/git/matching-refs/tags/" + quote(tag, safe="")))

    def delete_tag(self, tag):
        self.api(self.root + "/git/refs/tags/" + quote(tag, safe=""), "DELETE")

    def resolve_tag(self, tag):
        obj = self.api(self.root + "/git/ref/tags/" + quote(tag, safe=""))["object"]
        for _ in range(16):
            if obj["type"] == "commit":
                return obj["sha"]
            if obj["type"] != "tag":
                break
            obj = self.api(self.root + "/git/tags/" + obj["sha"])["object"]
        raise ValueError("Tag does not resolve to a commit")

    def upload(self, release, path):
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        if not token or any(char in token for char in '\r\n"'):
            raise ValueError("Invalid release credential")
        endpoint = release["upload_url"].split("{", 1)[0]
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or parsed.hostname != "uploads.github.com" or parsed.username or parsed.password:
            raise ValueError("Unexpected release upload host")
        url = endpoint + "?name=" + quote(path.name, safe="")
        command = ["curl", "--config", "-", "--fail", "--silent", "--show-error",
                   "--output", os.devnull, "--write-out", "%{http_code}", "--retry", "0",
                   "-X", "POST", "-H", "Accept: application/vnd.github+json",
                   "-H", "Content-Type: application/octet-stream",
                   "--data-binary", "@" + str(path), url]
        identifier = release["id"]
        expected_size = path.stat().st_size
        # Credentials travel through stdin, never in process arguments or exceptions.
        config = f'header = "Authorization: Bearer {token}"\n'
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            if attempt > 1:
                assert_draft_unchanged(self.get_release(identifier), release)
            print(f"Uploading {path.name} to draft {identifier} (attempt {attempt}/{_UPLOAD_ATTEMPTS}).", flush=True)
            result = subprocess.run(command, input=config, capture_output=True, text=True)
            status = result.stdout.strip()
            if result.returncode == 0 and status == "201":
                return

            response = f"HTTP {status}" if status and status != "000" else "no HTTP response"
            detail = result.stderr.strip().replace(token, "[redacted]")
            error = f"Asset upload failed for {path.name} ({response}, curl exit {result.returncode}). {detail}".rstrip()
            transient = status in _TRANSIENT_UPLOAD_STATUS or (
                status in {"", "000", "100", "201"} and result.returncode in _TRANSIENT_CURL_ERRORS
            )
            duplicate_after_retry = attempt > 1 and status == "422"
            if not (transient or duplicate_after_retry):
                raise RuntimeError(error)

            # A timed-out POST can still leave a complete asset or an empty starter.
            print(f"{error} Checking the provisional draft.", file=sys.stderr, flush=True)
            matches = [asset for asset in self.assets(identifier) if asset.get("name") == path.name]
            assert_draft_unchanged(self.get_release(identifier), release)
            if len(matches) > 1:
                raise ValueError(f"Multiple assets named {path.name} on draft {identifier}")
            asset = matches[0] if matches else None
            if asset and asset.get("state") == "uploaded" and asset.get("size") == expected_size:
                print(f"GitHub already stored {path.name} on draft {identifier}.", flush=True)
                return
            if attempt == _UPLOAD_ATTEMPTS or (duplicate_after_retry and asset is None):
                raise RuntimeError(error)
            if asset:
                if asset.get("state") != "starter" or asset.get("size") != 0:
                    raise ValueError(f"Unexpected asset state or size for {path.name} on draft {identifier}")
                asset_id = asset.get("id")
                if type(asset_id) is not int or asset_id <= 0:
                    raise ValueError(f"Invalid starter asset identifier for {path.name}")
                self.api(self.root + f"/releases/assets/{asset_id}", "DELETE")
            time.sleep(2 ** (attempt - 1))

    def assets(self, identifier):
        pages = self.api(self.root + f"/releases/{identifier}/assets?per_page=100", pages=True)
        return [item for page in pages for item in page]


def publish(client, *, mode, ref_name, release_tag, target, run_id, title, assets):
    original = client.releases()
    plan = replacement_plan(original, mode, ref_name, release_tag, run_id)
    if plan["skip"]:
        print("A newer branch preview already exists; keeping it unchanged.")
        return
    if mode == "tag" and not client.resolve_tag(release_tag).lower().startswith(target.lower()):
        raise ValueError("Tag moved or does not match the built commit")
    tag_exists = mode == "branch" and client.branch_tag_exists(release_tag)
    if tag_exists and not plan["owns_tag"]:
        raise ValueError("Refusing to move a synthetic tag without matching draft ownership")
    marker = f"<!-- github-draft-release-key: {mode}:{ref_name} -->"
    body = f"{marker}\n<!-- sessionprep-run: {run_id} -->\n\nSource: {ref_name} at {target}."
    candidate_tag = f"sessionprep-staging-{hashlib.sha256(release_tag.encode()).hexdigest()[:12]}-{run_id}-{uuid.uuid4().hex[:8]}"
    candidate = client.create_release(dict(tag_name=candidate_tag, target_commitish=target,
        name=title + " (preparing)", body=body + "\n<!-- sessionprep-candidate -->", draft=True,
        prerelease=mode == "branch"))
    identifier = candidate["id"]
    retirement_started = False
    try:
        for path in assets:
            client.upload(candidate, path)
        uploaded = client.assets(identifier)
        expected = {path.name: path.stat().st_size for path in assets}
        completed = {item["name"]: item["size"] for item in uploaded if item.get("state") == "uploaded"}
        if len(uploaded) != len(expected) or completed != expected:
            raise ValueError(f"Uploaded asset count/name/size mismatch or unfinished upload for release {identifier}")
        by_name = {item["name"]: item for item in uploaded}
        for path in assets:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if by_name[path.name].get("digest") != "sha256:" + digest.hexdigest():
                raise ValueError(f"Uploaded asset digest mismatch for {path.name}")
        # Another producer or a human may have acted during uploads.
        fresh = [item for item in client.releases() if item["id"] != identifier]
        fresh_plan = replacement_plan(fresh, mode, ref_name, release_tag, run_id)
        if fresh_plan["skip"]:
            raise ValueError("A newer preview appeared during upload; retaining it")
        if mode == "tag" and not client.resolve_tag(release_tag).lower().startswith(target.lower()):
            raise ValueError("Tag moved during upload; keeping prior releases unchanged")
        originals = {item["id"]: item for item in original}
        # Never delete an object that appeared/changed after the reviewed snapshot.
        if fresh_plan["delete"] != plan["delete"]:
            raise ValueError("Release set changed during upload; retry this build")
        assert_deletable(client.get_release(identifier), candidate, mode, ref_name)
        for old_id in plan["delete"]:
            assert_deletable(client.get_release(old_id), originals[old_id], mode, ref_name)
        for old_id in plan["delete"]:
            assert_deletable(client.get_release(old_id), originals[old_id], mode, ref_name)
            retirement_started = True
            client.delete_release(old_id)
        if tag_exists:
            if any(item.get("tag_name") == release_tag for item in client.releases()):
                raise ValueError("Synthetic tag acquired another release; refusing deletion")
            retirement_started = True
            client.delete_tag(release_tag)
        assert_deletable(client.get_release(identifier), candidate, mode, ref_name)
        final = client.update_release(identifier, dict(tag_name=release_tag, target_commitish=target,
                    name=title, body=body, draft=True, prerelease=mode == "branch"))
        print(f"Draft release ready: id={identifier} url={final.get('html_url', '')}")
    except Exception:
        if retirement_started:
            print(f"Verified candidate {identifier} retained after retirement began; review/retry promotion.", file=sys.stderr)
            raise
        # Failed preparation must preserve prior previews and published human work.
        # An ambiguous successful promotion is retained rather than deleted.
        try:
            current = client.get_release(identifier)
            assert_deletable(current, candidate, mode, ref_name)
            client.delete_release(identifier)
        except Exception:
            print(f"Candidate {identifier} changed or cleanup failed; left for review.", file=sys.stderr)
        if plan["delete"]:
            print("Draft replacement stopped before retiring any prior draft.", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tag = sub.add_parser("tag")
    tag.add_argument("ref_name")
    command = sub.add_parser("publish")
    command.add_argument("--mode", choices=("branch", "tag"), required=True)
    for name in ("ref-name", "release-tag", "target", "repo", "title"):
        command.add_argument("--" + name, required=True)
    command.add_argument("--producer-run-id", type=int, required=True)
    command.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "tag":
        print(branch_tag(args.ref_name))
    else:
        assets = sorted(path for path in args.assets.iterdir() if path.is_file())
        if not assets:
            raise ValueError("No staged release assets")
        publish(GitHub(args.repo), mode=args.mode, ref_name=args.ref_name, release_tag=args.release_tag,
                target=args.target, run_id=args.producer_run_id, title=args.title, assets=assets)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    try:
        main()
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
