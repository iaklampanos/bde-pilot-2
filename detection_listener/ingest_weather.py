import json
import requests
from operator import attrgetter
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input path')
    opts = parser.parse_args()
    getter = attrgetter('input')
    inp = getter(opts)
    
