import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_meta


class ReleaseMetadataTest(unittest.TestCase):
    def test_stable_tag_selects_release_and_preserves_version(self):
        for event in ("push", "workflow_dispatch"):
            self.assertEqual(
                {"tag": "v2.2.1", "is_release": "true", "is_prerelease": "false"},
                release_meta.resolve(event, "refs/tags/v2.2.1", "", ""),
            )

    def test_semver_prerelease_tags_use_release_identity(self):
        for tag in ("v2.2.1-alpha.1", "v2.2.1-beta.1", "v2.2.1-rc.1", "v2.2.1-preview.2+build.9"):
            with self.subTest(tag=tag):
                result = release_meta.resolve("push", "refs/tags/" + tag, "", "")
                self.assertEqual(tag, result["tag"])
                self.assertEqual("true", result["is_release"])
                self.assertEqual("true", result["is_prerelease"])

    def test_build_metadata_is_not_a_prerelease(self):
        result = release_meta.resolve("push", "refs/tags/v2.2.1+build.9", "", "")
        self.assertEqual("false", result["is_prerelease"])
        self.assertEqual("v2.2.1+build.9", result["tag"])

    def test_branch_at_a_release_tag_is_still_preview(self):
        for event in ("push", "workflow_dispatch"):
            for ref in ("refs/heads/v5", "refs/heads/v2.2.1"):
                result = release_meta.resolve(event, ref, "v2.2.1-0-gabcdef1", "abcdef1")
                self.assertEqual("false", result["is_release"])
                self.assertEqual("v2.2.1-ci.0-gabcdef1", result["tag"])

    def test_pull_request_never_selects_release(self):
        for ref in ("refs/pull/516/merge", "refs/tags/v2.2.1"):
            self.assertEqual("false", release_meta.resolve("pull_request", ref, "", "abcdef1")["is_release"])

    def test_invalid_tag_fails_before_building(self):
        for tag in ("vnext", "v2.2", "v02.2.1", "v2.2.1-beta.01", "v2.2.1-beta..1", "v2.2.1\nother=value"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release_meta.resolve("push", "refs/tags/" + tag, "", "")

    def test_preview_keeps_git_distance_and_semver_metadata(self):
        version = release_meta.preview_version("v2.2.1-beta.1+build.9-66-gabcdef1", "abcdef1")
        self.assertEqual("v2.2.1-beta.1-ci.66-gabcdef1+build.9", version)
        self.assertIsNotNone(release_meta.SEMVER_TAG.fullmatch(version))

    def test_preview_without_usable_tag_has_valid_fallback(self):
        for description in ("", "vnext-4-gabcdef1"):
            self.assertEqual("v0.0.0-ci.0-gabcdef1", release_meta.preview_version(description, "abcdef1"))


if __name__ == "__main__":
    unittest.main()
