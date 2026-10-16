"""Capture and manually box colour examples from the same camera used for sorting."""

import argparse

import cv2

from .camera_source import FrameSource
from .dataset import COLOR_CLASSES, init_dataset, save_sample


def annotate(frame):
    """Freeze a frame and collect hand-drawn boxes; no HSV-generated training labels."""
    preview = frame.copy()
    annotations = []
    window = 'Draw object box: ENTER accept, ESC finish boxes'
    try:
        while True:
            x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
            if w == 0 or h == 0:
                return annotations
            prompt = preview.copy()
            cv2.putText(prompt, '1 red | 2 green | 3 blue | 4 yellow | ESC discard box',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow(window, prompt)
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                if ord('1') <= key <= ord('4'):
                    cls = key - ord('1')
                    annotations.append((cls, x, y, w, h))
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(preview, COLOR_CLASSES[cls], (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    break
    finally:
        cv2.destroyWindow(window)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--camera', default='/dev/video0')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--out', default='datasets/colors')
    ap.add_argument('--split', choices=['train', 'val'], required=True)
    args = ap.parse_args()
    config = init_dataset(args.out)
    print(f'Dataset: {config}\nSPACE: freeze and label every object; B: save empty background; Q: quit')
    cam = FrameSource(args.camera, args.width, args.height)
    latest = None
    try:
        while True:
            frame = cam.read()
            if frame is not None:
                latest = frame
            # A displayed old frame must not be saved again as if it were a new capture.
            if latest is not None:
                preview = latest.copy()
                cv2.putText(preview, f'{args.split}: SPACE label | B background | Q quit',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow('IntelPick dataset', preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key not in (ord(' '), ord('b')):
                continue
            fresh = cam.read()
            if fresh is None:
                fresh = frame
            if fresh is None:
                print('No fresh camera frame; wait for the stream to reconnect.')
                continue
            annotations = annotate(fresh) if key == ord(' ') else []
            if key == ord(' ') and not annotations:
                print('No boxes; sample skipped. B explicitly saves background.')
                continue
            path = save_sample(config.parent, args.split, fresh, annotations)
            print(f'Saved {path.name}: {len(annotations)} objects')
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
