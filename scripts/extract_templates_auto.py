import cv2
from pathlib import Path

from chesslab.detection.board_detector import BoardDetector
from chesslab.splitting.board_splitter import BoardSplitter
from chesslab.splitting.square_color import square_background


INPUT_IMAGE = "tests/templates/start_position.png"
OUTPUT_DIR = Path("src/chesslab/recognition/templates/green")

# mapping position initiale -> pièce
START_POSITION = {
    "a1": "wR", "b1": "wN", "c1": "wB", "d1": "wQ",
    "e1": "wK", "f1": "wB", "g1": "wN", "h1": "wR",

    "a2": "wP", "b2": "wP", "c2": "wP", "d2": "wP",
    "e2": "wP", "f2": "wP", "g2": "wP", "h2": "wP",

    "a8": "bR", "b8": "bN", "c8": "bB", "d8": "bQ",
    "e8": "bK", "f8": "bB", "g8": "bN", "h8": "bR",

    "a7": "bP", "b7": "bP", "c7": "bP", "d7": "bP",
    "e7": "bP", "f7": "bP", "g7": "bP", "h7": "bP",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading screenshot...")
    img = cv2.imread(INPUT_IMAGE)

    detector = BoardDetector()
    splitter = BoardSplitter()

    print("detecting board...")
    det = detector.detect(img)

    if not det.detected:
        raise RuntimeError("Board not detected")

    print("splitting squares...")
    split = splitter.split(img, det.board)

    templates = {}

    print("extracting templates...")

    for name, square in split.squares.items():

        if name not in START_POSITION:
            continue

        piece = START_POSITION[name]

        bg = square_background(square.file, square.rank)

        key = f"{piece}_{bg}"

        if key not in templates:
            templates[key] = square.image

    print(f"templates found: {len(templates)}")

    for key, img in templates.items():
        out = OUTPUT_DIR / f"{key}.png"
        cv2.imwrite(str(out), img)
        print("saved", out)

    print("done.")


if __name__ == "__main__":
    main()