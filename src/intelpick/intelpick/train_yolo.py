"""Validate and train a YOLO detection model for IntelPick's colour classes."""

import argparse
from pathlib import Path

from .dataset import validate_dataset


def training_options(args):
    return dict(data=str(Path(args.data).expanduser().resolve()), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=str(Path(args.project).expanduser().resolve()), name=args.name,
                seed=0, workers=args.workers,
                # Hue or channel changes can turn a red training object blue without
                # changing its label. Preserve hue/saturation; allow brightness variation.
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, bgr=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default='datasets/colors/data.yaml')
    ap.add_argument('--model', default='yolo26n.pt', help='initial pretrained detection weights')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='cpu', help='cpu or CUDA device, e.g. 0')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--project', default='runs/colors')
    ap.add_argument('--name', default='train')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0:
        ap.error('epochs, imgsz and batch must be positive; workers must be >= 0')
    try:
        summary = validate_dataset(args.data)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    for split, counts in summary.items():
        print(f'{split}: {counts["images"]} images; objects: {counts["objects"]}')
    if args.check_only:
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        ap.error('Ultralytics is missing; see docs/YOLO.md for installation')
    model = YOLO(args.model)
    if model.task != 'detect':
        ap.error('use object detection weights, not classification / segmentation / OBB weights')
    model.train(**training_options(args))
    best = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    print(f'Trained weights: {best.resolve()}')
    print('Inspect held-out validation predictions before running the real arm.')


if __name__ == '__main__':
    main()
