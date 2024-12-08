# Use MediaPipe vision tasks to classify all the pages in a TIFF scan
import argparse
from pathlib import Path
from pprint import pprint

import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageSequence
from tqdm import tqdm


MODEL_PATH = '../efficientnet_lite4_psworkbook_weighted.tflite'
BaseOptions = mp.tasks.BaseOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageClassifierOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

def main(workbook: Path) -> None:
    predictions = []
    # Outermost context is MediaPipe classifier
    with ImageClassifier.create_from_options(options) as classifier:
        with Image.open(workbook) as im:
            for i, page in enumerate(tqdm(ImageSequence.Iterator(im))):
                if im.format != 'RGB':
                    buffer = np.asarray(im.convert('RGB'))
                else:
                    buffer = np.asarray(im)
                mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=buffer)
                result = classifier.classify(mpi)
                top_category = result.classifications[0].categories[0]
                predictions.append(f"Page {i}: {top_category.category_name} ({top_category.score:.2f})")

    pprint(predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True, help='TIFF file containing scanned workbook'
    )
    args = parser.parse_args()
    workbook = Path(args.input)
    main(workbook)
