"""Class for loading the IAM dataset, which encompasses both paragraphs and lines, with associated utilities."""
import os
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ElementTree
import zipfile

from boltons.cacheutils import cachedproperty
import toml

from text_recognizer.datasets.base import Dataset, _download_raw_dataset


DATA_DIRNAME = Path(__file__).parents[2].resolve() / 'data'
RAW_DATA_DIRNAME = DATA_DIRNAME / 'raw' / 'iam'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'
EXTRACTED_DATASET_DIRNAME = RAW_DATA_DIRNAME / 'iamdb'

DOWNSAMPLE_FACTOR = 2  # If images were downsampled, the regions must also be.


class IamDataset(Dataset):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """
    def __init__(self):
        self.metadata = toml.load(METADATA_FILENAME)

    def load_or_generate_data(self):
        if not self.xml_filenames:
            self._download_iam()

    @property
    def xml_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / 'xml').glob('*.xml'))

    def _download_iam(self):
        curdir = os.getcwd()
        os.chdir(RAW_DATA_DIRNAME)
        _download_raw_dataset(self.metadata)
        _extract_raw_dataset(self.metadata)
        os.chdir(curdir)

    @cachedproperty
    def line_regions_by_name(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        return {
            filename.stem: _get_line_regions_from_xml_file(filename)
            for filename in self.xml_filenames
        }

    def __repr__(self):
        """Print info about the dataset."""
        return (
            'IAM Dataset\n'
            f'Num forms: {len(self.xml_filenames)}\n'
        )


def _extract_raw_dataset(metadata):
    print('Extracting IAM data')
    with zipfile.ZipFile(metadata['filename'], 'r') as zip_file:
        zip_file.extractall()


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall('handwritten-part/line')
    return [_get_line_region_from_xml_element(el) for el in xml_line_elements]


def _get_line_region_from_xml_element(xml_line) -> Dict[str, int]:
    """
    line (xml element): has x, y, width, and height attributes
    """
    word_elements = xml_line.findall('word/cmp')
    x1s = [int(el.attrib['x']) for el in word_elements]
    y1s = [int(el.attrib['y']) for el in word_elements]
    x2s = [int(el.attrib['x']) + int(el.attrib['width']) for el in word_elements]
    y2s = [int(el.attrib['y']) + int(el.attrib['height']) for el in word_elements]
    return {
        'x1': min(x1s) // DOWNSAMPLE_FACTOR,
        'y1': min(y1s) // DOWNSAMPLE_FACTOR,
        'x2': max(x2s) // DOWNSAMPLE_FACTOR,
        'y2': max(y2s) // DOWNSAMPLE_FACTOR
    }


def main():
    dataset = IamDataset()
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
