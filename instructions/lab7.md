# Lab 7: Data Labeling and Versioning

## Data labeling interface

Please now fully label at least one paragraph, and then we will sync up and review some results.

(Review results and discuss any differences in annotation and how they could be prevented.)

## metadata.toml

TODO describe structure by looking at the IAM one

To compute the SHA256 hash of a file, run `shasum -a 256 <filename>`.

## Download the data

We can update `metadata.toml` with a convenient script that compares the SHA-256 of the current file with the SHA-256 of the new file.
There is a convenience task script defined: `tasks/update_fsdl_paragraphs_metadata.sh`.
