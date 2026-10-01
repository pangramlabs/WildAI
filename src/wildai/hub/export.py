"""Export trained checkpoints as Hugging Face model folders, staged in the release layout (see ``wildai.hub.layout``).

Each model folder holds model.safetensors, config.json (architecture plus the model's catalog entry under ``release``),
generation_config.json, the tokenizer, the remote-code files and a model card. The repository root gets the card that
lists every model, the remote-code files and the tokenizer.

One model:
    python -m wildai.hub.export --model 268m-g072-ai-r0.5 --checkpoint path/to/model_003360.pt \
        --tokenizer path/to/tokenizer.pkl --out staging/
Many models (a CSV with columns name,checkpoint):
    python -m wildai.hub.export --manifest checkpoints.csv --tokenizer path/to/tokenizer.pkl --out staging/
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from wildai.hub import remote_code
from wildai.hub.card import CardSettings, index_card, model_card
from wildai.hub.catalog import Catalog
from wildai.hub.checkpoint import ExportDtype, NanochatCheckpoint
from wildai.hub.layout import ReleaseNaming
from wildai.hub.tokenizer import BOS, load_encoding, save_hf_tokenizer, to_hf_tokenizer
from wildai.license import legal_code
from wildai.training.model import GPTConfig

REMOTE_CODE_FILES = ("configuration_wildai.py", "modeling_wildai.py")
AUTO_MAP = {
    "AutoConfig": "configuration_wildai.WildAIConfig",
    "AutoModelForCausalLM": "modeling_wildai.WildAIForCausalLM",
}


def copy_remote_code(directory: Path) -> None:
    source = Path(remote_code.__file__).parent
    directory.mkdir(parents=True, exist_ok=True)
    for name in REMOTE_CODE_FILES:
        shutil.copyfile(source / name, directory / name)


@dataclass(frozen=True)
class Exporter:
    catalog: Catalog
    tokenizer: Tokenizer
    settings: CardSettings
    dtype: ExportDtype = ExportDtype.bfloat16

    def export(self, name: str, checkpoint_path: Path, root: Path) -> Path:
        """Write one model folder under ``root``; returns the folder."""
        entry = self.catalog.entry(name)
        checkpoint = NanochatCheckpoint.load(checkpoint_path)
        if checkpoint.config.n_layer != entry.run.depth:
            raise ValueError(f"{checkpoint_path} has {checkpoint.config.n_layer} layers but {name} has depth {entry.run.depth}")
        model = checkpoint.to_hf_model(self.dtype, bos_token_id=self.tokenizer.token_to_id(BOS))
        model.config.release = entry.release_metadata()
        model.config.auto_map = AUTO_MAP

        out = self.settings.naming.location(name).staging_dir(root)
        model.save_pretrained(out)
        save_hf_tokenizer(self.tokenizer, out, model_max_length=model.config.max_position_embeddings)
        copy_remote_code(out)
        (out / "README.md").write_text(model_card(entry, self.settings))
        (out / "LICENSE").write_text(legal_code())
        return out

    def write_repo_root(self, root: Path) -> None:
        """The repository root: the card listing every model, the remote-code files, the tokenizer and the license."""
        repo_dir = self.settings.naming.staging_dir(root)
        copy_remote_code(repo_dir)
        save_hf_tokenizer(self.tokenizer, repo_dir, model_max_length=GPTConfig.sequence_len)
        (repo_dir / "README.md").write_text(index_card(self.catalog.entries(), self.settings))
        (repo_dir / "LICENSE").write_text(legal_code())


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    with path.open() as f:
        return [(row["name"], Path(row["checkpoint"])) for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="public name of the model (results/models.csv)")
    target.add_argument("--manifest", type=Path, help="CSV with columns name,checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="model_<step>.pt of --model (meta_<step>.json beside it)")
    parser.add_argument("--names", nargs="*", help="with --manifest: export only these models")
    parser.add_argument("--tokenizer", type=Path, required=True, help="the training tokenizer.pkl")
    parser.add_argument("--out", type=Path, required=True, help="staging root; models land in <out>/<repo>/<name>/")
    parser.add_argument("--dtype", type=ExportDtype, default=ExportDtype.bfloat16, choices=list(ExportDtype))
    parser.add_argument("--org", default="pangram")
    parser.add_argument("--repo", default="WildAI-models", help="name of the model repository")
    parser.add_argument("--dataset", default="pangram/WildAI", help="Hub id of the training dataset, for the card metadata")
    parser.add_argument("--skip-existing", action="store_true", help="leave models whose folder already has weights")
    args = parser.parse_args()

    if args.model and args.checkpoint is None:
        parser.error("--model needs --checkpoint")
    jobs = [(args.model, args.checkpoint)] if args.model else read_manifest(args.manifest)
    if args.names:
        wanted = set(args.names)
        jobs = [job for job in jobs if job[0] in wanted]
        missing = wanted - {name for name, _ in jobs}
        if missing:
            parser.error(f"not in the manifest: {sorted(missing)}")

    settings = CardSettings(naming=ReleaseNaming(org=args.org, repo=args.repo), dataset=args.dataset)
    exporter = Exporter(Catalog.load(), to_hf_tokenizer(load_encoding(args.tokenizer)), settings, args.dtype)
    for i, (name, checkpoint) in enumerate(jobs, 1):
        if args.skip_existing and (settings.naming.location(name).staging_dir(args.out) / "model.safetensors").exists():
            print(f"[{i}/{len(jobs)}] {name}: exists, skipped")
            continue
        out = exporter.export(name, checkpoint, args.out)
        print(f"[{i}/{len(jobs)}] {name} -> {out}")
    exporter.write_repo_root(args.out)


if __name__ == "__main__":
    main()
