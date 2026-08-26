"""Export one CIFAR-10 test image for curl-based endpoint checks."""
from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets import CIFAR10


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="CIFAR-10 cache directory")
    parser.add_argument("--output", default="test_image.png", help="output image path")
    parser.add_argument("--index", type=int, default=0, help="test-set image index")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = CIFAR10(root=args.data_dir, train=False, download=True)
    image, label = dataset[args.index]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"wrote {output_path} label={dataset.classes[label]} index={args.index}")


if __name__ == "__main__":
    main()
