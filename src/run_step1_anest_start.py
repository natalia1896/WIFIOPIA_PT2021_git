"""Step1 starter for Photothrombosis Anest_start project.

Place this file next to common.py and step1_project_creation.py, then run:
    python run_step1_anest_start.py --config "H:\\Photothrombosis_analysis\\config_photothrombosis_anest_start_step4.yaml"

Step1 discovers raw recordings, builds the manifest and exports full-resolution
reference BMP images for Ilastik. Step2 requires Ilastik probability H5 exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import build_standard_paths, create_or_load_project, load_config
from step1_project_creation import (
    build_recording_manifest,
    save_manifest,
    save_reference_images_for_manifest,
)


def main() -> None:
    """Create or load the anesthesia-onset project, discover recordings, and export Step1 Ilastik reference images.

    Notes
    -----
    This launcher delegates to the existing Step1 manifest and reference-image
    functions. It is not the anesthesia-onset Hb analysis itself."""
    parser = argparse.ArgumentParser(description="Prepare Anest_start Step1 project + Ilastik references")
    parser.add_argument("--config", required=True, type=Path, help="Path to the new project YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    project = create_or_load_project(config, args.config)
    paths = build_standard_paths(config, project)
    manifest = build_recording_manifest(config, paths)

    if manifest.empty:
        raise RuntimeError(
            "No matching recordings. Check discovery.raw_root, BMP count, "
            "metadata.regex and discovery.metadata_filters."
        )
    manifest = save_reference_images_for_manifest(manifest, config, paths, show_progress=True)
    save_manifest(manifest, paths["manifest_path"], overwrite=True)

    print("\nStep1 completed")
    print("Project:", project["analysis_root"])
    print("Recordings:", len(manifest))
    print("Manifest:", paths["manifest_path"])
    print("Ilastik BMPs:", paths["reference_images_root"])
    print("Next: export matching *_Probabilities.h5 from Ilastik, then run Step2.")
    print("\nSelected recordings:")
    columns = [c for c in ["Anest_type", "mouse_id", "Anest_time", "stem", "img_dir"] if c in manifest.columns]
    print(manifest[columns].to_string(index=False))


if __name__ == "__main__":
    main()
