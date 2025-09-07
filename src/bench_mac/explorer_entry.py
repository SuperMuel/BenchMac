from __future__ import annotations

import sys


def main() -> None:
    from streamlit.web.cli import main as streamlit_main

    sys.argv = [
        "streamlit",
        "run",
        "explorer/app.py",
    ]
    streamlit_main()


if __name__ == "__main__":
    main()
