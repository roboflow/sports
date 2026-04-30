#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

cat << EOF

Place trained model weights in:
  $DIR/data/handball-player-detection.pt
  $DIR/data/handball-ball-detection.pt
  $DIR/data/handball-court-keypoint-detection.pt

Use the notebooks in $DIR/notebooks to train models from the Roboflow Universe
datasets referenced in README.md.
EOF
