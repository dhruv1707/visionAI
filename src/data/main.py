import os
import argparse
from data import Data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        default=os.environ.get('INPUT_DIR'),
                        help="Mounted folder in ECS with your videos")
    
    parser.add_argument('-output_dir',
                        default=os.environ.get('OUTPUT_DIR'),
                        help='Output directory you want to write to')
    
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    if not args.input_dir or not args.output_dir:
        raise Exception("Must pass --input_dir and --output_dir (or set INPUT_DIR/OUTPUT_DIR)")
    
    data = Data(args.input_dir)

    data.process_video(args.input_dir, args.output_dir)

