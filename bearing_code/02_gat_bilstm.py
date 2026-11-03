"""Train or evaluate the full GAT-BiLSTM model. See README for commands."""
import sys
from bearing.experiments import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.extend(["--suite", "full"])
    main()
