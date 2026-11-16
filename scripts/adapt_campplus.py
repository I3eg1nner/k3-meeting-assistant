"""Rename the vendor CAM++ input for the existing 3D-Speaker native runner.

Weights are unchanged. Original and adapted outputs must numerically agree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise SystemExit('Keep the original model unchanged')
    graph = onnx.load(args.source)
    assert len(graph.graph.input) == 1
    previous = graph.graph.input[0].name
    graph.graph.input[0].name = 'feature'
    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            if name == previous:
                node.input[i] = 'feature'
    for item in graph.graph.value_info:
        if item.name == previous:
            item.name = 'feature'
    onnx.checker.check_model(graph)
    pending = args.output.with_suffix('.pending.onnx')
    onnx.save(graph, pending)
    original = ort.InferenceSession(str(args.source), providers=['CPUExecutionProvider'])
    adapted = ort.InferenceSession(str(pending), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    for frames in (100, 150, 345):
        sample = rng.normal(size=(1, frames, 80)).astype(np.float32)
        a = original.run(None, {previous:sample})[0]
        b = adapted.run(None, {'feature':sample})[0]
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    pending.replace(args.output)
    manifest = {'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(), 'adapted_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(), 'change':f'input {previous} -> feature; weights unchanged', 'equivalence_samples':3}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))

if __name__ == '__main__':
    main()
