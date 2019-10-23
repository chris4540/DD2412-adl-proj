import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_width', type=int, required=True)
    parser.add_argument('--teacher_depth', type=int, required=True)
    parser.add_argument('--student_width', type=int, required=True)
    parser.add_argument('--student_depth', type=int, required=True)
    parser.add_argument('--teacher_weight_path', required=False)

    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

