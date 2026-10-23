"""Launch Microsoft GraphRAG with the Windows selector event loop."""

from __future__ import annotations

import asyncio
import runpy
import sys


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.argv = ["graphrag", "index", *sys.argv[1:]]
    runpy.run_module("graphrag", run_name="__main__")
