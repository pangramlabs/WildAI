"""Create the model repository on the Hugging Face Hub and upload a staging root written by ``wildai.hub.export``.

Without ``--execute`` it only prints the plan (repository, local folder, model count and size). Uploading needs a token
with write access to the organisation (``hf auth login`` or ``HF_TOKEN``). ``--visibility`` sets the repository's
visibility, also for one that exists already. The repository is added to the organisation's WildAI collection. Exports
can be uploaded in parts (for example one model size at a time): each upload adds its models to the repository.

    python -m wildai.hub.publish --staging staging/                         # plan only
    python -m wildai.hub.publish --staging staging/ --execute               # create a private repository and upload
    python -m wildai.hub.publish --staging staging/ --visibility public --execute
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from wildai.hub.collection import TITLE, add_to_collection
from wildai.hub.layout import ReleaseNaming


@dataclass(frozen=True)
class RepoUpload:
    repo_id: str
    folder: Path
    models: int
    bytes: int


def plan(staging: Path, naming: ReleaseNaming) -> RepoUpload:
    """The staged repository (``<staging>/<repo>``, holding the root files and one subfolder per model)."""
    folder = naming.staging_dir(staging)
    if not (folder / "README.md").exists():
        raise FileNotFoundError(f"{folder} has no model card; run wildai.hub.export first")
    models = sum(1 for p in folder.iterdir() if (p / "model.safetensors").exists())
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and ".cache" not in p.parts)
    return RepoUpload(repo_id=naming.repo_id, folder=folder, models=models, bytes=size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", type=Path, required=True, help="the --out directory of wildai.hub.export")
    parser.add_argument("--org", default="pangram")
    parser.add_argument("--repo", default="WildAI-models")
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    parser.add_argument("--collection", default=TITLE, help="title of the organisation collection the repository joins")
    parser.add_argument("--execute", action="store_true", help="create the repository and upload (default: print the plan)")
    args = parser.parse_args()

    u = plan(args.staging, ReleaseNaming(org=args.org, repo=args.repo))
    print(f"{u.repo_id}: {u.models} models, {u.bytes / 1e9:.1f} GB, {args.visibility}  <- {u.folder}")
    if not args.execute:
        return
    api = HfApi()
    private = args.visibility == "private"
    api.create_repo(u.repo_id, repo_type="model", private=private, exist_ok=True)
    # Resumable and split into many commits; safe to re-run after an interruption.
    api.upload_large_folder(repo_id=u.repo_id, folder_path=u.folder, repo_type="model")
    api.update_repo_settings(u.repo_id, private=private, repo_type="model")
    add_to_collection(api, args.org, u.repo_id, "model", title=args.collection, private=private)


if __name__ == "__main__":
    main()
