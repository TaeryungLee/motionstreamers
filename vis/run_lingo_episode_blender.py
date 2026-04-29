from __future__ import annotations

import sys

from run_episode_blender import main


if __name__ == "__main__":
    main(["--dataset", "lingo", *sys.argv[1:]])
