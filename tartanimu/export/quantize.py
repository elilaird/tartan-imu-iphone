"""Post-training quantization for CoreML models.

Supports INT8 weight quantization and FP16 activation for
optimal Neural Engine performance on iPhone.
"""

import argparse

try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
except ImportError:
    ct = None


def quantize_model(
    input_path: str,
    output_path: str | None = None,
    nbits: int = 16,
) -> "ct.models.MLModel":
    """Quantize a CoreML model.

    Args:
        input_path: path to .mlpackage
        output_path: path for quantized output (default: adds _q{nbits} suffix)
        nbits: quantization bits (8 or 16)
    Returns:
        Quantized CoreML model
    """
    if ct is None:
        raise ImportError("coremltools is required: pip install coremltools")

    model = ct.models.MLModel(input_path)

    if nbits == 16:
        # FP16 — usually already done at export, but can re-apply
        quantized = ct.models.neural_network.quantization_utils.quantize_weights(
            model, nbits=16
        )
    elif nbits == 8:
        # INT8 weight quantization
        quantized = ct.models.neural_network.quantization_utils.quantize_weights(
            model, nbits=8
        )
    else:
        raise ValueError(f"Unsupported nbits={nbits}. Use 8 or 16.")

    if output_path is None:
        output_path = input_path.replace(".mlpackage", f"_q{nbits}.mlpackage")

    quantized.save(output_path)
    print(f"Quantized model saved to {output_path}")
    return quantized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize CoreML model")
    parser.add_argument("input", help="Input .mlpackage path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--nbits", type=int, default=8, choices=[8, 16])
    args = parser.parse_args()

    quantize_model(args.input, args.output, args.nbits)
