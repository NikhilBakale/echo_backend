#!/bin/bash
# Post-build cleanup script
# Remove unnecessary files to reduce deployment size

echo "Starting post-build cleanup..."

# Remove Python cache files
find /home/site/wwwroot -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /home/site/wwwroot -type f -name "*.pyc" -delete 2>/dev/null || true
find /home/site/wwwroot -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove test files
find /home/site/wwwroot -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find /home/site/wwwroot -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Remove docs and examples from packages
find /home/site/wwwroot -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true
find /home/site/wwwroot -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true

# Keep only essential PyTorch files (remove test/benchmark files)
if [ -d "/home/site/wwwroot/antenv/lib/python3.11/site-packages/torch" ]; then
    cd /home/site/wwwroot/antenv/lib/python3.11/site-packages/torch
    rm -rf test/ || true
    rm -rf testing/ || true
fi

echo "Post-build cleanup completed"
echo "Disk usage after cleanup:"
du -sh /home/site/wwwroot 2>/dev/null || echo "Cannot calculate size"
