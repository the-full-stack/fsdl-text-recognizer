"""Class for loading our own FSDL Handwriting dataset, which encompasses both paragraphs and lines."""
import json

from boltons.cacheutils import cachedproperty
import toml

from text_recognizer.datasets.base import Dataset


RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'fsdl_handwriting'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'
PAGES_DIRNAME = RAW_DATA_DIRNAME / 'pages'


class FsdlHandwritingDataset(Dataset):
    """
    FSDL Handwriting dataset gathered in class.
    """
    def __init__(self):
        self.metadata = toml.load(METADATA_FILENAME)
        with open(RAW_DATA_DIRNAME / self.metadata['filename']) as f:
            self.data = [json.loads(line) for line in f.readlines()]
        from IPython import embed
        embed()

    def load_or_generate_data(self):
        if not self.page_filenames:
            self._download_pages()

    @property
    def page_filenames(self):
        return list(PAGES_DIRNAME.glob('*.jpg'))

    def _download_pages(self):
        pass

    @cachedproperty
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        pass

    @cachedproperty
    def line_content_by_id(self):
        """Return a dict from name of image to a list of strings."""
        pass

    def __repr__(self):
        return (
            'FSDH Handwriting Dataset\n'
            f'Num pages: {len(self.data)}\n'
        )


def main():
    dataset = FsdlHandwritingDataset()
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
