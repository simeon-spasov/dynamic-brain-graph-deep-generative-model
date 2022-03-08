import argparse

from src.experiment import run
from src.utils import load_config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dynamic graph generative model")
    parser.add_argument("-c", "--config_file", type=str, default="model", help="yaml file with parameters for model, training, testing, and output")
    args = parser.parse_args()
    config = load_config(args.config_file)                    
    return config

def main(args):
    run(args)
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
