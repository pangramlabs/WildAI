"""Create the dataset repository on the Hugging Face Hub and upload a release folder.

The repository is created private by default under the ``pangram`` organization (``hub`` in
``configs/data/release.yaml``); ``--public``, ``--org`` and ``--name`` override the config. Gating is set only in the
config, because the dataset card must carry the matching access-request fields. Uploading needs a token with write
access to the organization (``HF_TOKEN`` or ``huggingface-cli login``).

    python -m wildai.data.release.push --release-dir release
    python -m wildai.data.release.push --release-dir release --org my-org --name WildAI-sample
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

from wildai.data.config import default_config, load_config
from wildai.data.release.config import HubSettings, ReleaseConfig
from wildai.hub.collection import CollectionApi, add_to_collection


class HubApi(CollectionApi, Protocol):
    """The part of ``huggingface_hub.HfApi`` the upload uses."""

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> object: ...

    def update_repo_settings(self, repo_id: str, *, gated: str, repo_type: str) -> object: ...

    def upload_large_folder(self, repo_id: str, folder_path: str, *, repo_type: str) -> object: ...


def push(release_dir: Path, hub: HubSettings, api: HubApi) -> str:
    if not (release_dir / "README.md").exists():
        raise FileNotFoundError(f"{release_dir} has no dataset card; run wildai.data.release.card first")
    api.create_repo(hub.repo_id, repo_type="dataset", private=hub.private, exist_ok=True)
    if hub.gated:
        api.update_repo_settings(hub.repo_id, gated="manual", repo_type="dataset")
    api.upload_large_folder(hub.repo_id, str(release_dir), repo_type="dataset")
    add_to_collection(api, hub.org, hub.repo_id, "dataset", title=hub.collection, private=hub.private)
    return f"https://huggingface.co/datasets/{hub.repo_id}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("release.yaml"))
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--org")
    parser.add_argument("--name")
    parser.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    args = parser.parse_args(argv)
    hub = load_config(args.config, ReleaseConfig).hub
    updates = {k: v for k, v in {"org": args.org, "name": args.name}.items() if v}
    if args.public:
        updates["private"] = False
    hub = hub.model_copy(update=updates)
    from huggingface_hub import HfApi

    print(push(args.release_dir, hub, HfApi()))


if __name__ == "__main__":
    main()
