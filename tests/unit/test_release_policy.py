import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_policy", ROOT / "packaging/github/release_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def release(identifier, ref, *, draft=True, run=10, tag="0.3.5"):
    return dict(id=identifier, draft=draft, tag_name=tag,
                body=f"<!-- github-draft-release-key: branch:{ref} -->\n<!-- sessionprep-run: {run} -->")


def test_branch_slug_collisions_have_distinct_tags():
    assert policy.branch_tag("feature/audio") != policy.branch_tag("feature-audio")
    assert policy.branch_tag("Feature/audio") != policy.branch_tag("feature/audio")


def test_cleanup_is_branch_owned_not_version_owned():
    tag = policy.branch_tag("main")
    plan = policy.replacement_plan([release(1,"main",tag=tag),release(2,"release/0.3.5"),
                                    dict(id=3,draft=True,tag_name="0.3.5",body="")],
                                   "branch", "main", tag, 11)
    assert plan["delete"] == [1]


def test_older_run_cannot_replace_newer_preview():
    tag = policy.branch_tag("main")
    plan = policy.replacement_plan([release(1,"main",run=20,tag=tag)],"branch","main",tag,19)
    assert plan["skip"] is True and plan["delete"] == []


@pytest.mark.parametrize("mode", ["tag","branch"])
def test_published_releases_are_protected(mode):
    tag = policy.branch_tag("main") if mode == "branch" else "0.3.5"
    with pytest.raises(ValueError, match="published"):
        policy.replacement_plan([release(1,"main",draft=False,tag=tag)],mode,"main",tag,11)


def test_tag_replacement_does_not_delete_branch_draft():
    assert policy.replacement_plan([release(1,"0.3.5",tag="branch-build-old"),
                                    dict(id=2,draft=True,tag_name="0.3.5",body="")],
                                   "tag","0.3.5","0.3.5",11)["delete"] == [2]


def test_unowned_tag_collision_fails_before_mutation():
    with pytest.raises(ValueError, match="ownership"):
        policy.replacement_plan([dict(id=1,draft=True,tag_name="custom",body="")],
                                "branch","main","custom",11)


class FakeGitHub:
    def __init__(self, releases, *, upload_failure=False, publish_old=False, publish_candidate=False, tag_commits=None):
        import copy
        self.items = {r["id"]: copy.deepcopy(r) for r in releases}
        self.deleted = []
        self.uploaded = []
        self.upload_failure = upload_failure
        self.publish_old = publish_old
        self.publish_candidate = publish_candidate
        self.tag_commits = list(tag_commits or ["a" * 40])

    def releases(self):
        import copy
        return copy.deepcopy(list(self.items.values()))

    def branch_tag_exists(self, tag):
        return False

    def resolve_tag(self, tag):
        return self.tag_commits.pop(0) if len(self.tag_commits) > 1 else self.tag_commits[0]

    def get_release(self, identifier):
        import copy
        if (identifier == 1 and self.publish_old) or (identifier == 999 and self.publish_candidate):
            self.items[identifier]["draft"] = False
        return copy.deepcopy(self.items[identifier])

    def create_release(self, payload):
        import copy
        self.items[999] = dict(payload, id=999, upload_url="unused")
        return copy.deepcopy(self.items[999])

    def upload(self, release, path):
        if self.upload_failure:
            raise RuntimeError("simulated upload failure")
        self.uploaded.append(dict(name=path.name, size=path.stat().st_size))

    def assets(self, identifier):
        return self.uploaded

    def delete_release(self, identifier):
        assert self.items[identifier]["draft"] is True
        self.deleted.append(identifier)
        del self.items[identifier]

    def update_release(self, identifier, payload):
        self.items[identifier].update(payload)
        return self.items[identifier]


def invoke(client, tmp_path, mode="branch"):
    asset = tmp_path / "installer.exe"
    asset.write_bytes(b"installer")
    policy.publish(client, mode=mode, ref_name="main" if mode=="branch" else "0.3.5",
                   release_tag=policy.branch_tag("main") if mode=="branch" else "0.3.5",
                   target="a"*40, run_id=11, title="Preview", assets=[asset])


def test_upload_failure_keeps_previous_preview(tmp_path):
    client = FakeGitHub([release(1,"main",tag=policy.branch_tag("main"))], upload_failure=True)
    with pytest.raises(RuntimeError, match="upload failure"):
        invoke(client,tmp_path)
    assert 1 in client.items and client.deleted == [999]


def test_human_publication_before_delete_is_preserved(tmp_path):
    client = FakeGitHub([release(1,"main",tag=policy.branch_tag("main"))], publish_old=True)
    with pytest.raises(ValueError, match="published"):
        invoke(client,tmp_path)
    assert client.items[1]["draft"] is False and 1 not in client.deleted


def test_human_publication_of_failed_candidate_is_preserved(tmp_path):
    client = FakeGitHub([release(1,"main",tag=policy.branch_tag("main"))], upload_failure=True, publish_candidate=True)
    with pytest.raises(RuntimeError, match="upload failure"):
        invoke(client,tmp_path)
    assert 1 in client.items and client.items[999]["draft"] is False and not client.deleted


def test_tag_moving_during_upload_keeps_prior_release(tmp_path):
    client = FakeGitHub([dict(id=1, draft=True,tag_name="0.3.5",body="old")],tag_commits=["a"*40,"b"*40])
    with pytest.raises(ValueError, match="Tag moved"):
        invoke(client,tmp_path,"tag")
    assert 1 in client.items and 1 not in client.deleted


def test_success_promotes_only_complete_candidate(tmp_path):
    client = FakeGitHub([release(1,"main",tag=policy.branch_tag("main"))])
    invoke(client,tmp_path)
    assert client.deleted == [1]
    assert client.items[999]["tag_name"] == policy.branch_tag("main")
    assert client.items[999]["draft"] is True and client.items[999]["prerelease"] is True


@pytest.mark.parametrize("annotated", [True,False])
def test_remote_tag_resolution_peels_annotated_tags(annotated):
    client = policy.GitHub("owner/repo")
    def api(path):
        if "/git/ref/" in path:
            return {"object":{"type":"tag" if annotated else "commit", "sha":"b"*40 if annotated else "a"*40}}
        assert path.endswith("b"*40)
        return {"object":{"type":"commit","sha":"a"*40}}
    client.api = api
    assert client.resolve_tag("v0.3.5") == "a"*40


def test_promotion_failure_after_retirement_retains_verified_candidate(tmp_path):
    client=FakeGitHub([release(1,"main",tag=policy.branch_tag("main"))])
    def fail(*args):
        raise RuntimeError("promotion failed")
    client.update_release=fail
    with pytest.raises(RuntimeError,match="promotion failed"):
        invoke(client,tmp_path)
    assert client.deleted==[1] and 999 in client.items and client.items[999]["draft"] is True


def test_candidate_publication_before_retirement_keeps_previous_draft(tmp_path):
    client=FakeGitHub([release(1,"main",tag=policy.branch_tag("main"))],publish_candidate=True)
    with pytest.raises(ValueError,match="published"):
        invoke(client,tmp_path)
    assert 1 in client.items and not client.deleted
