import json
import math
import sys
from pathlib import Path
import tkinter as tk

MAX_WIDTH = 1400
MAX_HEIGHT = 900

if len(sys.argv) != 2:
    print("Usage: python3 scripts/annotate_board_tk.py path/to/image.png")
    sys.exit(1)

image_path = Path(sys.argv[1])
if not image_path.exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

points = []

root = tk.Tk()
root.title("Clique TOP-LEFT puis BOTTOM-RIGHT du plateau")

orig_img = tk.PhotoImage(file=str(image_path))

# facteur entier de réduction pour que toute l'image tienne dans la fenêtre
factor_w = math.ceil(orig_img.width() / MAX_WIDTH)
factor_h = math.ceil(orig_img.height() / MAX_HEIGHT)
scale_factor = max(1, factor_w, factor_h)

if scale_factor > 1:
    display_img = orig_img.subsample(scale_factor, scale_factor)
else:
    display_img = orig_img

canvas = tk.Canvas(
    root,
    width=display_img.width(),
    height=display_img.height(),
    cursor="cross"
)
canvas.pack()

canvas.create_image(0, 0, anchor="nw", image=display_img)

status = tk.Label(root, text="Clique le coin haut-gauche du plateau", anchor="w")
status.pack(fill="x")


def on_click(event):
    # coordonnées dans l'image originale
    x = event.x * scale_factor
    y = event.y * scale_factor

    points.append((x, y))
    print(f"Click {len(points)}: ({x}, {y})")

    r = 4
    canvas.create_oval(
        event.x - r, event.y - r, event.x + r, event.y + r,
        outline="red", width=2
    )

    if len(points) == 1:
        status.config(text="Clique maintenant le coin bas-droit du plateau")

    elif len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]

        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # rectangle visuel sur l'image affichée
        canvas.create_rectangle(
            x / scale_factor,
            y / scale_factor,
            (x + width) / scale_factor,
            (y + height) / scale_factor,
            outline="lime",
            width=2
        )

        result = {
            "image": image_path.name,
            "board": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            },
        }

        print("\nDetected board:")
        print(f"x={int(x)}, y={int(y)}, width={int(width)}, height={int(height)}")

        print("\nJSON line:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        status.config(
            text="Rectangle calculé. Copie le JSON dans ton manifest, puis ferme la fenêtre."
        )


canvas.bind("<Button-1>", on_click)
root.mainloop()