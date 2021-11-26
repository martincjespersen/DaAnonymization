"""Console script for textprivacy."""
import argparse
import sys

from textprivacy import TextAnonymizer, TextPseudonymizer


def main():
    """
    Commandline version of TextPrivacy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="Text to be masked")
    parser.add_argument(
        "-m",
        "--masking",
        type=str,
        default="anonymizer",
        const="anonymizer",
        nargs="?",
        choices=["pseudonymizer", "anonymizer"],
        help="Masking technique to apply",
    )
    args = parser.parse_args()

    if args.masking == "anonymizer":
        mask_transformer = TextAnonymizer([args.input])
    else:
        mask_transformer = TextPseudonymizer([args.input])

    masked_corpus = mask_transformer.mask_corpus()
    print(masked_corpus[-1])


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
