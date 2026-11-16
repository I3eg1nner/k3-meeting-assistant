#!/usr/bin/env python3
"""Export the official 3D-Speaker CAM++ checkpoint to dynamic ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakerlab", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.speakerlab))
    import onnx
    import torch
    from speakerlab.models.campplus.DTDNN import CAMPPlus

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, 345, 80)
    with torch.no_grad():
        expected = model(sample).numpy()
    torch.onnx.export(
        model,
        sample,
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["feature"],
        output_names=["embedding"],
        dynamic_axes={
            "feature": {0: "batch_size", 1: "frame_num"},
            "embedding": {0: "batch_size"},
        },
        dynamo=False,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    print(
        f"exported {args.output} ({args.output.stat().st_size} bytes), "
        f"reference embedding shape={expected.shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
