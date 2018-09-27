import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    for fn in os.listdir(args.dir):
        base, ext = os.path.splitext(fn)
        new_name = "".join([base, ".1", ext])
        src = os.path.join(args.dir, fn)
        tgt = os.path.join(args.dir, new_name)
        print(src)
        print(tgt)
        os.rename(src, tgt)

if __name__ == "__main__":
    main()
