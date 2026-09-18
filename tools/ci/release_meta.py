"""Resolve the shared platform version and release channel from the workflow ref."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


NUMBER = r"(?:0|[1-9][0-9]*)"
IDENTIFIER = rf"(?:{NUMBER}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER_TAG = re.compile(
    rf"v{NUMBER}\.{NUMBER}\.{NUMBER}"
    rf"(?:-(?P<prerelease>{IDENTIFIER}(?:\.{IDENTIFIER})*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def release_tag(event: str, ref: str) -> str | None:
    if event not in {"push", "workflow_dispatch"} or not ref.startswith("refs/tags/"):
        return None
    tag = ref.removeprefix("refs/tags/")
    if SEMVER_TAG.fullmatch(tag) is None:
        raise ValueError(f"Release tags must be v-prefixed SemVer: {tag!r}")
    return tag


def preview_version(description: str, short_sha: str) -> str:
    # --long preserves the distance suffix even when HEAD is exactly on a tag.
    match = re.fullmatch(r"(.+)-([0-9]+)-g([0-9a-f]+)", description)
    if match and SEMVER_TAG.fullmatch(match[1]):
        base, distance, revision = match.groups()
    else:
        base, distance, revision = "v0.0.0", "0", short_sha
    version, separator, metadata = base.partition("+")
    return f"{version}-ci.{distance}-g{revision}" + (separator + metadata if separator else "")


def resolve(event: str, ref: str, description: str, short_sha: str) -> dict[str, str]:
    tag = release_tag(event, ref)
    if tag is not None:
        return {
            "tag": tag,
            "is_release": "true",
            "is_prerelease": str(SEMVER_TAG.fullmatch(tag)["prerelease"] is not None).lower(),
        }
    return {"tag": preview_version(description, short_sha), "is_release": "false", "is_prerelease": "false"}


def main() -> None:
    event, ref = os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"]
    # Validate tag refs before any build or signing job is allowed to run.
    tag = release_tag(event, ref)
    description, short_sha = "", ""
    if tag is None:
        description = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "v*", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        short_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    values = resolve(event, ref, description, short_sha)
    output = "".join(f"{key}={value}\n" for key, value in values.items())
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as target:
        target.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
