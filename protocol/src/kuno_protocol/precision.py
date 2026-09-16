"""Weight precision recipes: how a hardware class's `precision` becomes loaded weights, and the
weights identity (`model_digest`) that step transcripts carry and validators pin.

A profile variant is `<profile_id>@<hardware_class>`. Each variant loads exactly one recipe, chosen
from the class's `precision` ("bf16" when the class declares none) and the profile's `variant`
("fast", "pro", "dfr"). A quantized class therefore has its own weights digest (the owner-signed
manifest's `model_digests["<profile>@<class>"]`), its own golden set (`kuno_validator.golden`,
per profile and class) and its own tolerance calibration entries (`kuno_protocol.tolerance`, per
profile, miner class and executor class), as VERIFIED_MODE.md requires.

Weights identity (normative):

    model_digest = SHA-256("kuno/v1/weights\\n" | canonical_json({
        "identity": recipe.identity(),                   # recipe id, precision, subfolder, per-component method
        "files": [{"path", "size", "sha256"}, ...],      # every file the recipe reads, sorted by path
    }))

The method is part of the identity because an fp8 cast of the same bf16 files is different weights.
Memory figures in the recipes are estimates until `worker/scripts/benchmark_ltx_quantized.py`
measures them; they never enter the digest.

Recipes are data (precision_recipes.json). torch is never imported here: the worker's loader
(`kuno_worker.backends.quantized`) applies a recipe, and `kuno-devkit weights-digest` computes the
digest the owner publishes, on any machine holding the weights.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .canonical import canonical_json, sha256_hex
from .profiles import HardwareClass, ModelProfile

RECIPES_FILE = "precision_recipes.json"
WEIGHTS_TAG = b"kuno/v1/weights\n"
DEFAULT_PRECISION = "bf16"
_CHUNK = 16 << 20


class PrecisionError(ValueError):
    """No recipe serves this profile on this class, or the weights on disk are not the pinned ones."""


class ComponentPrecision(BaseModel):
    """How one pipeline component is stored and computed."""

    model_config = ConfigDict(extra="forbid")

    # "none": loaded as compute_dtype. "layerwise-cast": weights stored as storage_dtype and upcast per
    # layer for compute (diffusers enable_layerwise_casting; ltx-pipelines' fp8-cast does the same plain
    # cast). "torchao": a torchao AOBaseConfig applied at load (weight-only int8 here).
    method: Literal["none", "layerwise-cast", "torchao"] = "none"
    storage_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    torchao_config: str | None = None
    torchao_kwargs: dict[str, int | str | bool] = Field(default_factory=dict)
    skip_modules_pattern: list[str] = Field(default_factory=list)


class MemoryModel(BaseModel):
    """GPU memory in GiB, as a linear model of the stage with the most latent tokens.

    peak(tokens) = weights on the GPU for the offload mode + activation_fixed_gib
                   + activation_gib_per_10k_tokens × tokens / 10 000 + overhead_gib

    The prompt enhancer never runs inside a render, so without offload it waits in host RAM: a render's weights on the GPU
    are every component but `prompt_enhancer`, and writing text (an enhanced prompt, a plan) peaks at all of them.
    """

    model_config = ConfigDict(extra="forbid")

    # Keys: transformer, text_encoder, prompt_enhancer, other (VAEs, vocoder, connectors, upsamplers).
    components_gib: dict[str, float]
    # With group offload: the largest block group on the GPU at once, plus its prefetched neighbour.
    group_onload_gib: float = 1.0
    activation_fixed_gib: float
    activation_gib_per_10k_tokens: float
    overhead_gib: float = 1.5
    measured: bool = False
    basis: str

    @property
    def weights_gib(self) -> float:
        return sum(self.components_gib.values())


class PrecisionRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    variants: list[str]
    precision: str
    runtime: str
    transformer_subfolder: str
    # Files and directories (relative to the models dir) the loader reads; all of them are hashed.
    include: list[str]
    components: dict[str, ComponentPrecision]
    min_compute_capability: tuple[int, int] | None = None
    packages: dict[str, str] = Field(default_factory=dict)
    # Per-file SHA-256 published upstream, where it is (LTX-2.5 is gated: none are visible today).
    pinned_sha256: dict[str, str] = Field(default_factory=dict)
    memory: MemoryModel
    sources: list[str] = Field(default_factory=list)

    def identity(self) -> dict:
        return {
            "recipe": self.id,
            "precision": self.precision,
            "transformer_subfolder": self.transformer_subfolder,
            "components": {name: spec.model_dump(mode="json") for name, spec in sorted(self.components.items())},
        }


class WeightFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int
    sha256: str | None = None


class WeightsCheck(BaseModel):
    """What a load verified: the recipe, the digest it computed (or trusted) and the files it covers."""

    recipe_id: str
    mode: Literal["full", "size"]
    model_digest: str
    files: list[WeightFile]


def variant_id(profile_id: str, hardware_class: str) -> str:
    return f"{profile_id}@{hardware_class}"


def class_precision(hardware: HardwareClass | None) -> str:
    return (hardware.precision if hardware is not None else None) or DEFAULT_PRECISION


@lru_cache(maxsize=1)
def load_recipes() -> dict[str, PrecisionRecipe]:
    raw = json.loads(resources.files(__package__).joinpath(RECIPES_FILE).read_text())
    return {r["id"]: PrecisionRecipe.model_validate(r) for r in raw["recipes"]}


def select_recipe(
    profile: ModelProfile, hardware_class: str | None, recipes: dict[str, PrecisionRecipe] | None = None
) -> tuple[PrecisionRecipe, HardwareClass | None]:
    """The recipe for this profile on this class. No class (performance mode) means bf16."""
    recipes = load_recipes() if recipes is None else recipes
    hardware = None
    if hardware_class is not None:
        hardware = profile.verified.hardware_class(hardware_class) if profile.verified else None
        if hardware is None:
            raise PrecisionError(f"{profile.id} does not list hardware class {hardware_class}; pick one of its verified classes")
    precision = class_precision(hardware)
    matches = [r for r in recipes.values() if r.family == profile.family and r.precision == precision and profile.variant in r.variants]
    if not matches:
        where = f" on {hardware_class}" if hardware_class else ""
        raise PrecisionError(f"no {precision} weights recipe for {profile.id} (variant {profile.variant}){where}")
    if len(matches) > 1:
        raise PrecisionError(f"ambiguous recipes for {profile.id} at {precision}: {', '.join(r.id for r in matches)}")
    return matches[0], hardware


# ------------------------------------------------------------------ weights on disk


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def weight_files(root: Path, recipe: PrecisionRecipe) -> list[str]:
    """Every file the recipe reads, as sorted POSIX paths. Hidden files (download caches) are skipped."""
    root = Path(root)
    found: set[str] = set()
    for entry in recipe.include:
        path = root / entry
        if path.is_file():
            found.add(Path(entry).as_posix())
        elif path.is_dir():
            for item in path.rglob("*"):
                relative = item.relative_to(root)
                if item.is_file() and not any(part.startswith(".") for part in relative.parts):
                    found.add(relative.as_posix())
        else:
            raise PrecisionError(f"{recipe.id} needs {entry} under {root}, which is missing")
    if not found:
        raise PrecisionError(f"{recipe.id} found no weight files under {root}")
    return sorted(found)


def weights_digest(recipe: PrecisionRecipe, files: list[WeightFile]) -> str:
    if any(f.sha256 is None for f in files):
        raise PrecisionError("a weights digest needs every file's SHA-256")
    listing = [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in sorted(files, key=lambda f: f.path)]
    return sha256_hex(WEIGHTS_TAG + canonical_json({"identity": recipe.identity(), "files": listing}))


def verify_weights(
    root: Path,
    recipe: PrecisionRecipe,
    *,
    expected_digest: str | None = None,
    mode: Literal["full", "size"] = "full",
    allow_unpinned: bool = False,
    hasher=sha256_file,
) -> WeightsCheck:
    """Refuses weights that are not the pinned ones, naming what differs.

    full  hashes every file: a published per-file pin must match, the digest must equal
          `expected_digest` (the manifest's) when given, and without it every file must be pinned
          unless `allow_unpinned` (development only).
    size  only checks the files exist and trusts `expected_digest`, which it requires. Use it only
          where something else guarantees content: a dm-verity mount whose root hash RTMR3 binds.
    """
    root = Path(root)
    paths = weight_files(root, recipe)
    if mode == "size":
        if not expected_digest:
            raise PrecisionError("weights verification 'size' needs the manifest's model digest (KUNO_MODEL_DIGEST)")
        files = [WeightFile(path=p, size=(root / p).stat().st_size) for p in paths]
        if any(f.size == 0 for f in files):
            raise PrecisionError(f"empty weight file: {next(f.path for f in files if f.size == 0)}")
        return WeightsCheck(recipe_id=recipe.id, mode="size", model_digest=expected_digest, files=files)
    if mode != "full":
        raise PrecisionError(f"unknown weights verification mode {mode!r}; use full or size")
    files = []
    for path in paths:
        sha = hasher(root / path)
        pinned = recipe.pinned_sha256.get(path)
        if pinned is not None and pinned != sha:
            raise PrecisionError(f"{path} does not match its published SHA-256 ({recipe.id}): re-download it")
        files.append(WeightFile(path=path, size=(root / path).stat().st_size, sha256=sha))
    missing = [p for p in recipe.pinned_sha256 if p not in paths]
    if missing:
        raise PrecisionError(f"{recipe.id} pins files that are missing: {', '.join(missing[:3])}")
    digest = weights_digest(recipe, files)
    if expected_digest is not None and digest != expected_digest:
        raise PrecisionError(
            f"weights under {root} hash to {digest[:16]}…, not the manifest's {expected_digest[:16]}… for {recipe.id}: "
            "wrong files, a different download revision, or the wrong precision class"
        )
    if expected_digest is None and not allow_unpinned:
        unpinned = [p for p in paths if p not in recipe.pinned_sha256]
        if unpinned:
            raise PrecisionError(
                f"{len(unpinned)} weight file(s) of {recipe.id} have no published pin and no manifest digest was given: "
                "set KUNO_MODEL_DIGEST to the owner-signed manifest's model_digests entry for this profile and class"
            )
    return WeightsCheck(recipe_id=recipe.id, mode="full", model_digest=digest, files=files)
