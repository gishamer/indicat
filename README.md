# indicat

This repository contains the indicative action item annotations and scripts to map the annotations to the transcripts of the ICSI and ISL corpus.

## Getting Started

### Prepare the data

1. Download the [ICSI original MRT format transcripts with documentation](http://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_original_transcripts.zip). Make sure you really download the original format, and not the core annotations. Unzip it and move the directory containing the `transcript` directory to the `data` directory of this repository.
2. Download the [ISL Corpus](https://ca.talkbank.org/data/ISL.zip). Unzip it and move the `ISL` directory to the `data` directory of this repository.
3. Ensure the resulting directory looks like this:

```bash
data
├── ICSI_original_transcripts
├── ISL
└── indicat.csv
```

4. Install dependencies with `pip install -r requirements.txt`
5. Execute with `python main.py`: This creates a file named `indicat_corpus.csv` in the repositories root directory, which contains the annotations and the corresponding text.

If you want to keep the corpora in a different location, you can change the respective entries in `configs.yaml`
