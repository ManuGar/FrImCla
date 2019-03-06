from __future__ import absolute_import
from . import fullAnalysis
import argparse
import sys

def main():
    arg1 = sys.argv[1]
    fullAnalysis.fullAnalysis(arg1)