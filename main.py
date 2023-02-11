from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
import re
from typing import Callable, Iterator
import xml.etree.ElementTree as ET

import pandas as pd
import yaml


JOIN_COLUMNS = ['transcript', 'start_time', 'end_time', 'participant']


@dataclass
class AidaAnnotation:
    name: str
    annotator: str


@dataclass
class Utterance:
    start_time: float
    end_time: float
    participant: str
    text: str
    annotations: list[AidaAnnotation]


@dataclass
class Transcript:
    name: str
    utterances: list[Utterance]


class CorpusName(Enum):
    ICSI = 'ICSI'
    ISL = 'ISL'


@dataclass
class TidyUtterance:
    transcript: str
    start_time: float
    end_time: float
    participant: str
    text: str


def compose_create_transcript(parser: Callable[[Path], list[Utterance]]) -> Callable[[Path], Transcript]:
    def create_transcript(file_path: Path) -> Transcript:
        return Transcript(
            file_path.stem,
            parser(file_path)
        )

    return create_transcript


def parse_corpus(
        files_to_exclude: list[str],
        file_ending: str,
        parser: Callable[[Path], list[Utterance]],
        folder_path: Path
) -> list[Transcript]:
    create_transcript = compose_create_transcript(parser)
    return sorted(
        [create_transcript(file_path) for file_path in Path(folder_path).glob(f'*.{file_ending}')
         if file_path.name not in files_to_exclude], key=lambda x: x.name)


def assemble_text(segment: ET.Element) -> str:
    if segment.text and segment.text.strip():
        text = f'{segment.text} {segment.tail or ""}'
    elif segment.tail and segment.tail.strip():
        text = f'<{segment.tag}> {segment.tail}'
    else:
        text = '<' + segment.tag + '>'
    return re.sub(r'\s+', ' ', text).strip()


###########################
#      Parse ICSI         #
###########################


def parse_icsi_text(segment: ET.Element) -> str:
    if not segment:
        return assemble_text(segment)
    else:
        return re.sub(
            r'\s+', ' ',
            (segment.text or '')
            + ' '.join(parse_icsi_text(sub_segment) for sub_segment in segment)
        ).strip()


def parse_icsi_utterance(segment: ET.Element) -> Utterance:
    return Utterance(
        float(segment.attrib['StartTime']),
        float(segment.attrib['EndTime']),
        segment.attrib.get('Participant', 'NotHuman'),
        parse_icsi_text(segment),
        list()
    )


def parse_icsi_transcript(path: Path) -> list[Utterance]:
    return [parse_icsi_utterance(segment) for segment in ET.parse(path).getroot().find('Transcript').findall('Segment')
            if 'DigitTask' not in segment.keys()]


parse_icsi = partial(
    parse_corpus,
    ['preambles.mrt', 'readme'],
    'mrt',
    parse_icsi_transcript
)

###########################
#       Parse ISL         #
###########################


begins_with_short_name = re.compile(r'\*[A-Z]{3}[\s\S]+')
isl_naming_pattern = re.compile(r'\*[A-Z]{3}:')
whitespaces_and_newline = re.compile(r'[\n\t\s]+')
timing_info = r'\x15.+\x15'
brackets = re.compile(r'[<>]|\[.*?\]|\(.*?\)')
non_verbal_annotations = re.compile(r'(?<=&=)\w*')
at_after_word = re.compile(r'(?<=\w)@\w+')
isl_annotations = re.compile(r'(/|\+|-)')


def extract_timing(utterance: str) -> tuple[float, float]:
    start_time, end_time = [float(re.search('[0-9]+', time).group()) / 1_000
                            for time in re.search(timing_info, utterance).group().split('_')] \
        if re.search(timing_info, utterance) else (-1, -1)
    return start_time, end_time


def parse_isl_utterance(utterance: tuple[str, str]) -> Utterance:
    start_time, end_time = extract_timing(utterance[1])
    return Utterance(
        start_time=start_time,
        end_time=end_time,
        participant=utterance[0][1:-1],
        text=extract_text(utterance[1]),
        annotations=list()
    )


def reformat_non_verbal_annotations(utterance):
    annotation = re.finditer(non_verbal_annotations, utterance)
    if annotation:
        for match in annotation:
            utterance = utterance[:match.span()[0] - 2] \
                + '<' + utterance[match.span()[0]:match.span()[1]] + '>' \
                + utterance[match.span()[1]:]
    return utterance


def extract_text(utterance: str) -> str:
    utterance = re.sub(at_after_word, '', utterance)
    utterance = re.sub(brackets, '', utterance)
    utterance = re.sub(isl_annotations, '', utterance)
    utterance = re.sub(timing_info, '', utterance).strip()
    utterance = reformat_non_verbal_annotations(utterance)
    utterance = re.sub(whitespaces_and_newline, ' ', utterance)
    return utterance


def extract_isl_utterances(path: Path) -> Iterator[tuple[str, str]]:
    with open(path) as f:
        transcript = f.read()
        utterances = re.findall(begins_with_short_name, transcript)[0]
        utterances = re.sub(whitespaces_and_newline, ' ', utterances)
        utterance_speakers = re.findall(isl_naming_pattern, utterances)
        utterances = isl_naming_pattern.split(utterances)
        utterances = [utterance.strip() for utterance in utterances[1:]]
        assert len(utterances) == len(utterance_speakers)

    return zip(utterance_speakers, utterances)


def parse_isl_transcript(path: Path) -> list[Utterance]:
    return [parse_isl_utterance(utterance) for utterance in extract_isl_utterances(path)]


parse_isl = partial(parse_corpus,
                    ['0metadata.cdc', 'm060.cha'],
                    'cha',
                    parse_isl_transcript)


###########################
#      Merge Corpora      #
###########################


def utterance_to_row(transcript_name: str, utterance: Utterance):
    annotations = {
        annotation.annotator: annotation.name for annotation in utterance.annotations}
    return TidyUtterance(
        transcript_name,
        utterance.start_time,
        utterance.end_time,
        utterance.participant,
        utterance.text
    )


def convert_corpus_to_dataframe(corpus: list[Transcript]) -> pd.DataFrame:
    """
    Converts the properties of the contained utterances to columns in a
    dataframe, and adds the name of the transcript to each column
    :param corpus: contains parsed utterances with added annotations
    :return: a dataframe with one utterance per row
    """
    return pd.DataFrame([
        utterance_to_row(transcript.name, utterance)
        for transcript in corpus for utterance in transcript.utterances
    ])


def create_corpus_df(icsi_path: Path, isl_path: Path):
    corpus_df = convert_corpus_to_dataframe(
        parse_icsi(icsi_path) + parse_isl(isl_path)
    )
    return corpus_df.convert_dtypes()


tag = r'<.*?>'   
multi_space = r' +'

def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.replace(tag, '', regex=True)
    df['text'] = df['text'].str.replace(multi_space, ' ', regex=True).str.strip()
    df = df[df['text'].str.match('\W*')]

    return df


def merge_corpus_with_annotations():
    with open('configs.yaml') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    df_corpus = create_corpus_df(
        configs['icsi_path'],
        configs['isl_path']
    )
    df_corpus = clean_corpus(df_corpus)

    df_annotations = pd.read_csv(configs['annotations_path'])
    df_annotations = df_annotations.convert_dtypes()

    df = pd.merge(
        df_annotations,
        df_corpus,
        how="inner",
        on=JOIN_COLUMNS,
        suffixes=("_annot", "_corpus"),
    )
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df.to_csv(configs['target_path'], index=False)

    print(f'Corpus written to: {Path.cwd() / configs["target_path"]}')



if __name__ == '__main__':
    merge_corpus_with_annotations()

