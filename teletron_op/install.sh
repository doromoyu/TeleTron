#!/bin/bash
set -e 

echo "Installing fused_adaln..."
(cd ./fused_adaln && python adaln_setup.py install)

echo "Installing fused_rmsnorm..."
(cd ./fused_rmsnorm && python rms_setup.py install)

echo "All installations complete."