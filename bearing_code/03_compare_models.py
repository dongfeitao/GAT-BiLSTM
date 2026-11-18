"""SVM and eight neural-model comparison, including the proposed model."""
import sys
from bearing.experiments import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.extend(["--suite", "baseline"])
    main()
