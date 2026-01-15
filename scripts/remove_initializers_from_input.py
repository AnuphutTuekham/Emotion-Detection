"""Remove initializers from ONNX graph inputs.

This resolves warnings like:
  Initializer ... appears in graph inputs and will not be treated as constant value/weight.

Usage:
  python scripts/remove_initializers_from_input.py --in public/models/onnx_model.onnx --out public/models/onnx_model.fixed.onnx
"""
import argparse
import onnx


def remove_initializers(input_path: str, output_path: str):
    model = onnx.load(input_path)
    init_names = {init.name for init in model.graph.initializer}

    # Filter graph.input to exclude any that are actually initializers
    inputs = [i for i in model.graph.input if i.name not in init_names]
    removed = [i.name for i in model.graph.input if i.name in init_names]
    model.graph.input.clear()
    model.graph.input.extend(inputs)

    onnx.save(model, output_path)
    print(f"Saved fixed model to: {output_path}")
    if removed:
        print("Removed the following inputs (they were initializers):")
        for n in removed:
            print(" -", n)
    else:
        print("No initializers found in graph inputs.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inpath", required=True)
    p.add_argument("--out", dest="outpath", required=True)
    args = p.parse_args()
    remove_initializers(args.inpath, args.outpath)


if __name__ == "__main__":
    main()
