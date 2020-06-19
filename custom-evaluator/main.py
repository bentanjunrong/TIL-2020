from utils.json_to_pd import json_to_pd
from utils.evaluator import evaluate, calc_MAP
import argparse
import sys

def pars_args(args):
    parser = argparse.ArgumentParser(description='calculate MAP from JSONs')
    parser.add_argument('--truth_path',   help='Path to the JSON file you are getting your ground truth values from',default='val.json')
    parser.add_argument('--ans_path',   help='Path to the JSON file containing answers', default='ans.json')
    return parser.parse_args(args)

def main():
    args = pars_args(sys.argv[1:])
    truth_annotations, ans_annotations = json_to_pd(args.truth_path, args.ans_path)
    recall, precision = evaluate(truth_annotations,ans_annotations)
    print(calc_MAP(precision,recall))

def notebook_func(truth_path='val.json', ans_path='ans.json'):
    truth_annotations, ans_annotations = json_to_pd(truth_path, ans_path)
    return evaluate(truth_annotations,ans_annotations)
if __name__ == '__main__':
    main()
