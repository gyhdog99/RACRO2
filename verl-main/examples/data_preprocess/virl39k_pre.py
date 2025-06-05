import os
import pandas as pd
from datasets import Dataset, Features, Value, Sequence, Image
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-parquet', default='path/to/src_parquet')
    parser.add_argument('--tgt-dir', default='path/to/tgt_dir')

    args = parser.parse_args()

    # Get the directory where the parquet file is located
    root_dir = os.path.dirname(os.path.abspath(args.src_parquet))

    # Load the parquet file
    df = pd.read_parquet(args.src_parquet)


    df = df[df['source'] != 'DeepScaleR']
    df = df.reset_index(drop=True)

    # Convert relative image paths to absolute paths (images stored in 'images' folder)
    def to_absolute_paths(image_paths):
        return [os.path.join(root_dir, path) for path in image_paths]

    df["image"] = df["image"].apply(to_absolute_paths)

    # Now convert to Hugging Face dataset with proper feature specification
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "PassRate_32BTrained": Value("float32"),
        "PassRate_7BBase": Value("float32"),
        "category": Value("string"),
        "source": Value("string"),
        "qid": Value("string"),
        "image": Sequence(Image())  # Multiple images per row
    })

    # Create the Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df, features=features)

# Save it to disk
hf_dataset.save_to_disk(args.tgt_dir)
