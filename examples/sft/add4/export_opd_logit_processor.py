"""Run in the student SGLang environment; write the built-in exclusion processor."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(DisallowedTokensLogitsProcessor.to_str() + '\n')


if __name__ == '__main__':
    main()
