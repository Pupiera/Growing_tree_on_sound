"""
Data preparation.

Download: https://www.ortolang.fr/market/corpora/cefc-orfeo

Author
-----
Adrien PUPIER (based on template from Titouan Parcollet)
"""
import os
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip

logger = logging.getLogger(__name__)


def prepare_cefcOrfeo(
    data_folder,
    save_folder,
    train_tsv_file=None,
    dev_tsv_file=None,
    test_tsv_file=None,
    accented_letters=False,
    skip_prep=False,
):

    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download:

    Arguments
    ---------
    data_folder : str
        Path to the folder where the pre-processed
        (via create_tsv.py and splitwav.py) Cefc-orfeo dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    train_tsv_file : str, optional
        Path to the Train Common Voice .tsv file (cs)
    dev_tsv_file : str, optional
        Path to the Dev Common Voice .tsv file (cs)
    test_tsv_file : str, optional
        Path to the Test Common Voice .tsv file (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    skip_prep: bool
        If True, skip data preparation.
    """
    if skip_prep:
        return

    if train_tsv_file is None:
        train_tsv_file = data_folder + "/train.tsv"
    else:
        train_tsv_file = train_tsv_file

    if dev_tsv_file is None:
        dev_tsv_file = data_folder + "/dev.tsv"
    else:
        dev_tsv_file = dev_tsv_file

    if test_tsv_file is None:
        test_tsv_file = data_folder + "/test.tsv"
    else:
        test_tsv_file = test_tsv_file

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = save_folder + "/train.csv"
    save_csv_dev = save_folder + "/dev.csv"
    save_csv_test = save_folder + "/test.csv"

    # If csv already exists, we skip the data preparation
    if skip(save_csv_train, save_csv_dev, save_csv_test):

        msg = "%s already exists, skipping data preparation!" % (save_csv_train)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_dev)
        logger.info(msg)

        msg = "%s already exists, skipping data preparation!" % (save_csv_test)
        logger.info(msg)

        return
    check_cefcOrfeo_file(data_folder)

    if train_tsv_file is not None:

        create_csv(
            train_tsv_file,
            save_csv_train,
            data_folder,
            accented_letters,
        )

    # Creating csv file for dev data
    if dev_tsv_file is not None:

        create_csv(
            dev_tsv_file,
            save_csv_dev,
            data_folder,
            accented_letters,
        )

    # Creating csv file for test data
    if test_tsv_file is not None:

        create_csv(
            test_tsv_file,
            save_csv_test,
            data_folder,
            accented_letters,
        )


def skip(save_csv_train, save_csv_dev, save_csv_test):
    """
    Detects if the Common Voice data preparation has been already done.

    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking folders and save options
    skip = False

    if (
        os.path.isfile(save_csv_train)
        and os.path.isfile(save_csv_dev)
        and os.path.isfile(save_csv_test)
    ):
        skip = True

    return skip


def create_csv(
    orig_tsv_file, csv_file, data_folder, accented_letters=False, language="en"
):

    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.

    Returns
    -------
    None
    """

    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "wrd", "pos", "gov", "dep", "start_word", "end_word"]]

    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):

        line = line[0]
        splited = line.split("\t")
        # Path is at indice 1 in cefc-orfeo tsv files.
        file = splited[1]
        # all file are stored in "nameofcorpus_clips"
        try:
            folder = file.split("-")[1]
        except IndexError as e:
            print(line)
            raise IndexError(e)
        wav_path = (
            data_folder
            + "/"
            + folder
            + "/"
            + folder
            + "_clips/"
            + file.replace(" ", "")
        )
        file_name = wav_path.split(".")[-2].split("/")[-1]
        s_id = line.split("\t")[0].replace(" ", "")
        if os.path.isfile(wav_path):
            info = torchaudio.info(wav_path)
        else:
            # print(wav_path)
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        duration = info.num_frames / info.sample_rate
        total_duration += duration

        # Getting transcript
        words = splited[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # !! Language specific cleaning !!
        words = words.replace("#\t", "NNAAMMEE\t")
        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()
        words = words.replace("'", " ")
        words = words.replace("’", " ")
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")
        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            continue

        pos = splited[3].upper()
        # Harmonizing data (VINF and VNF are the same label and N and NOM too)
        pos = pos.replace("VINF", "VNF").replace(" N ", " NOM ")
        gov = splited[4].upper()
        dep = splited[5].replace("\n", "").upper()

        start_word = splited[6]
        end_word = splited[7]
        # Composition of the csv_line
        csv_line = [s_id, str(duration), wav_path, str(words), pos, gov, dep, start_word, end_word]
        csv_lines.append(csv_line)

    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def strip_accents(text):

    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")

    return str(text)


def check_cefcOrfeo_file(data_folder):
    """
    Check if the data folder actually contains the preprocessed Cefc orfeo dataset.

    If not, raises an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """
    files_str = "/cfpb/cfpb_clips"

    # checking first folder
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the orfeo dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


if __name__ == "__main__":

    user_folder = "/home/getalp/data/ASR_data/FR/CORPUS_AUDIO"
    data_folder = user_folder + "/cefc-orfeo_v.1.5_december2021/11/cleaned_v2"
    save_folder = "orfeosave/orfeo_cleaned_v2"
    train_tsv_file = (
        user_folder + "/cefc-orfeo_v.1.5_december2021/11/cleaned_v2/mixed_train.tsv"
    )
    dev_tsv_file = (
        user_folder + "/cefc-orfeo_v.1.5_december2021/11/cleaned_v2/gold_valid.tsv"
    )
    test_tsv_file = (
        user_folder + "/cefc-orfeo_v.1.5_december2021/11/cleaned_v2/gold_test.tsv"
    )
    accented_letters = True
    duration_threshold = 10
    prepare_cefcOrfeo(
        data_folder,
        save_folder,
        train_tsv_file,
        dev_tsv_file,
        test_tsv_file,
        accented_letters,
    )
