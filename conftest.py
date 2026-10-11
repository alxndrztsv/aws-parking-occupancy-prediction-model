"""
Placed at the project root so that pytest puts it on sys.path and
`import src...` resolves from tests/ without installing the package.

It also makes the Lambda modules importable as top-level modules (that is how the
Lambda runtime loads them) and gives boto3 a region, because several modules build
an S3 client at import time and boto3 raises NoRegionError without one.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "lambda_functions"))

os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
