"""Reusable Tkinter dialog helpers for ChessLab GUI."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

from chesslab.gui import theme


@dataclass(frozen=True)
class ScrollableDialogLayout:
    dialog: tk.Toplevel
    canvas: tk.Canvas
    scrollbar: tk.Scrollbar
    container: tk.Frame
    window_id: int


def create_scrollable_dialog(
    parent: tk.Widget,
    *,
    title: str,
    padx: int = 24,
    pady: int = 20,
) -> ScrollableDialogLayout:
    """Create a modal dialog with a vertically scrollable content container."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.configure(bg=theme.BG)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(True, True)

    outer = tk.Frame(dialog, bg=theme.BG)
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(
        outer,
        bg=theme.BG,
        highlightthickness=0,
        bd=0,
    )
    scrollbar = tk.Scrollbar(
        outer,
        orient="vertical",
        command=canvas.yview,
        highlightthickness=0,
        bd=0,
        troughcolor=theme.BG_SECONDARY,
        activebackground=theme.BTN_HOVER,
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    container = tk.Frame(canvas, bg=theme.BG, padx=padx, pady=pady)
    window_id = canvas.create_window((0, 0), window=container, anchor="nw")

    container.bind(
        "<Configure>",
        lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.bind(
        "<Configure>",
        lambda event: canvas.itemconfigure(window_id, width=event.width),
    )

    return ScrollableDialogLayout(
        dialog=dialog,
        canvas=canvas,
        scrollbar=scrollbar,
        container=container,
        window_id=window_id,
    )


def finalize_scrollable_dialog(
    parent: tk.Widget,
    layout: ScrollableDialogLayout,
) -> None:
    """Fit a scrollable dialog to the screen and center it over *parent*."""
    dialog = layout.dialog
    dialog.update_idletasks()

    max_width = max(420, dialog.winfo_screenwidth() - 80)
    max_height = max(320, dialog.winfo_screenheight() - 80)
    width = min(dialog.winfo_reqwidth(), max_width)
    height = min(dialog.winfo_reqheight(), max_height)

    dialog.maxsize(max_width, max_height)

    x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    dialog.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
    layout.canvas.configure(scrollregion=layout.canvas.bbox("all"))
    layout.canvas.yview_moveto(0.0)


def center_dialog(parent: tk.Widget, dialog: tk.Toplevel) -> None:
    """Center an already-created dialog over *parent*."""
    dialog.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() - dialog.winfo_width()) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - dialog.winfo_height()) // 2
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
