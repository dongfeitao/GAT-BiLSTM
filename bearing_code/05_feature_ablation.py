"""Both domains, time-only and frequency-only controlled ablations."""
import sys
from bearing.experiments import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.extend(["--suite", "feature"])
    main()
