#!/usr/bin/env python3
import json
import argparse

def filter_scores(input_file, output_file, min_score=4):
    """Reads a JSONL file, filters lines with a 'score' greater than min_score,
    and writes selected keys plus a 'source' field to an output file.
    """
    # Define which keys to include in the filtered data.
    keys = ['prompt_id', 'prompt', 'completion']
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Warning: Skipping an invalid JSON line.")
                continue
            # Check if the 'score' exceeds the threshold
            if obj.get('score', 0) > min_score:
                # Create a new dictionary with only the selected keys
                new_obj = {k: obj.get(k) for k in keys}
                new_obj['source'] = 'generated'
                outfile.write(json.dumps(new_obj) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Filter a JSONL file to output entries with a score above a specified threshold."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--min_score", type=int, default=7,
        help="Minimum score threshold (default is 4)."
    )
    
    args = parser.parse_args()
    filter_scores(args.input, args.output, args.min_score)

if __name__ == '__main__':
    main()
