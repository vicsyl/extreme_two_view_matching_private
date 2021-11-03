import argparse
from pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(prog='pipeline')
    args = parser.parse_args()
    pipeline, _ = Pipeline.configure("config.txt", args)
    print(pipeline.output_dir)


if __name__ == "__main__":
    main()
