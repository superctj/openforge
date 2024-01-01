"""Most of this module is from the D3L codebase.

https://github.com/alex-bogatu/d3l/blob/main/d3l/indexing/feature_extraction/values/fasttext_embedding_transformer.py
https://github.com/alex-bogatu/d3l/blob/main/d3l/indexing/feature_extraction/schema/qgram_transformer.py
https://github.com/alex-bogatu/d3l/blob/main/d3l/utils/constants.py
https://github.com/alex-bogatu/d3l/blob/main/d3l/utils/functions.py
"""

import gzip
import os
import random
import re
import shutil
import unicodedata

from typing import Iterable, List, Optional, Set
from urllib.request import urlopen

import numpy as np

from fasttext import load_model
from nltk.corpus import stopwords
from openforge.ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from sklearn.feature_extraction.text import TfidfVectorizer


STOPWORDS = set(stopwords.words('english'))

SYMBPATT = r"\@" + re.escape(
    u"".join(chr(i) for i in range(0xFFFF) if unicodedata.category(chr(i)) == "Sc")
)
PUNCTPATT = r"\!\"\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\[\\\]\^\_\`\{\|\}\~"

LOWALPHA = re.compile(r"[a-z]([a-z\-])*")
UPPALPHA = re.compile(r"[A-Z]([A-Z\-\.])*")
CAPALPHA = re.compile(r"[A-Z][a-z]([a-z\-])*")

POSDEC = re.compile(r"\+?[0-9]+(,[0-9]+)*\.[0-9]+")
NEGDEC = re.compile(r"\-[0-9]+(,[0-9]+)*\.[0-9]+")
POSINT = re.compile(r"\+?[0-9]+(,[0-9]+)*")
NEGINT = re.compile(r"\-[0-9]+(,[0-9]+)*")

PUNCT = re.compile(r"[" + PUNCTPATT + r"]+")
SYMB = re.compile(r"[" + SYMBPATT + r"]+", re.UNICODE)
WHITE = re.compile(r"\s+")

ALPHANUM = re.compile(r"(?:[0-9]+[a-zA-Z]|[a-zA-Z]+[0-9])[a-zA-Z0-9]*")
NUMSYMB = re.compile(
    r"(?=.*[0-9,\.])(?=.*[" + SYMBPATT + r"]+)([0-9" + SYMBPATT + r"]+)", re.UNICODE
)

FASTTEXTURL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
GLOVEURL = "http://nlp.stanford.edu/data/"

CONSTANT = 1e-9
VALUE_SIGNATURE_ATTEMPTS = 100


def shingles(value: str) -> Iterable[str]:
    """
    Generate multi-word tokens delimited by punctuation.
        Parameters
        ----------
        value : str
            The value to shingle.

        Returns
        -------
        Iterable[str]
            A generator of shingles.
    """

    delimiterPattern = re.compile(r"[^\w\s\-_@&]+")
    for shingle in delimiterPattern.split(value):
        yield re.sub(r"\s+", " ", shingle.strip().lower())


class QGramTransformer:
    def __init__(self, qgram_size: Optional[int] = None):
        """
        This object performs qgram extraction.
        Parameters
        ----------
        qgram_size : Optional[int]
            The default qgram size.
        """
        self._qgram_size = qgram_size

    @property
    def qgram_size(self) -> Optional[int]:
        return self._qgram_size

    def transform(
        self, input_string: str, qgram_size: Optional[int] = None
    ) -> List[str]:
        """
        Generate a collection of qgrams of configured size from the given string.
        Parameters
        ----------
        input_string : str
            The input string to transform.
        qgram_size : Optional[int]
            The size of each qgram.
            If None, the upper-level size (passed when the object was created) will be used.

        Returns
        -------
        List[str]
            A collection of qgrams of the given string.

        """

        if qgram_size is None and self._qgram_size is None:
            raise ValueError(
                "Expected a qgram_size in this call or at the object level."
            )

        elif qgram_size is None:
            qgram_size = self._qgram_size

        qgrams = []
        for word in re.split(r"\W+", input_string.lower().strip()):
            word = word.strip()
            if len(word) < 1:
                continue
            if len(word) <= qgram_size:
                qgrams.append(word)
                continue
            for i in range((len(word) - qgram_size) + 1):
                qgrams.append(word[i : i + qgram_size])
        return qgrams


class FasttextTransformer:
    def __init__(
        self,
        token_pattern: str = r"(?u)\b\w\w+\b",
        max_df: float = 0.5,
        stop_words: Iterable[str] = STOPWORDS,
        embedding_model_lang="en",
        cache_dir: Optional[str] = None,
    ):
        """
        Instantiate a new embedding-based transformer
        Parameters
        ----------
        token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's TfidfVectorizer default.
        max_df : float
            Percentage of values the token can appear in before it is ignored.
        stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        embedding_model_lang : str
            The embedding model language.
        cache_dir : Optional[str]
            An exising directory path where the model will be stored.
            If not given, the current working directory will be used.
        """

        self._token_pattern = token_pattern
        self._max_df = max_df
        self._stop_words = stop_words
        self._embedding_model_lang = embedding_model_lang
        self._cache_dir = (
            cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
        )

        self._embedding_model = self.get_embedding_model(
            overwrite=False,
        )

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "_embedding_model"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = self.get_embedding_model(overwrite=False)

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def _download_fasttext(self, model_file_name: str, chunk_size: int = 2 ** 13):
        """
        Download pre-trained common-crawl vectors from fastText's website
        https://fasttext.cc/docs/en/crawl-vectors.html

        Parameters
        ----------
        model_file_name : str
            The model file name to download.
        chunk_size : int
            The Fasttext models are commonly large - several GBs.
            The disk writing will therefore be made in chunks.

        Returns
        -------

        """

        url = FASTTEXTURL + model_file_name
        print("Downloading %s" % url)
        response = urlopen(url)

        downloaded = 0
        write_file_name = (
            os.path.join(self._cache_dir, model_file_name)
            if self._cache_dir is not None
            else model_file_name
        )
        download_file_name = write_file_name + ".part"
        with open(download_file_name, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                downloaded += len(chunk)
                if not chunk:
                    break
                f.write(chunk)
                # print("{} downloaded ...".format(downloaded))

        os.rename(download_file_name, write_file_name)

    def _download_model(self, if_exists: str = "strict"):
        """
        Download the pre-trained model file.
        Parameters
        ----------
        if_exists : str
            Supported values:
                - *ignore*: The model will not be downloaded
                - *strict*: This is the defaul. The model will be downloaded only if it does not exist at the *cache_dir*.
                - *overwrite*: The model will be downloaded even if it already exists at the *cache_dir*.

        Returns
        -------

        """

        base_file_name = "cc.%s.300.bin" % self._embedding_model_lang
        file_name = (
            os.path.join(self._cache_dir, base_file_name)
            if self._cache_dir is not None
            else base_file_name
        )
        gz_file_name = "%s.gz" % base_file_name

        if os.path.isfile(file_name):
            if if_exists == "ignore":
                return file_name
            elif if_exists == "strict":
                print("File exists. Use --overwrite to download anyway.")
                return file_name
            elif if_exists == "overwrite":
                pass

        absolute_gz_file_name = (
            os.path.join(self._cache_dir, gz_file_name)
            if self._cache_dir is not None
            else gz_file_name
        )
        if not os.path.isfile(absolute_gz_file_name):
            self._download_fasttext(gz_file_name)

        with gzip.open(absolute_gz_file_name, "rb") as f:
            with open(file_name, "wb") as f_out:
                shutil.copyfileobj(f, f_out)

        """Cleanup"""
        if os.path.isfile(absolute_gz_file_name):
            os.remove(absolute_gz_file_name)

        return file_name

    def get_embedding_model(
        self,
        overwrite: bool = False,
    ):
        """
        Download, if not exists, and load the pretrained FastText embedding model in the working directory.
        Note that the default gzipped English Common Crawl FastText model has 4.2 GB
        and its unzipped version has 6.7 GB.
        Parameters
        ----------
        overwrite : bool
            If True overwrites the model if exists.

        Returns
        -------

        """
        if_exists = "strict" if not overwrite else "overwrite"

        model_file = self._download_model(if_exists=if_exists)
        embedding_model = load_model(model_file)
        return embedding_model

    def get_embedding_dimension(self) -> int:
        """
        Retrieve the embedding dimensions of the underlying model.
        Returns
        -------
        int
            The dimensions of each embedding
        """
        return self._embedding_model.get_dimension()

    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding of the given word.
        If the word is out of vocabulary a zero vector is returned.
        Parameters
        ----------
        word : str
            The word to retrieve the vector for.

        Returns
        -------
        np.ndarray
            A vector of float numbers.
        """
        vector = self._embedding_model.get_word_vector(
            str(word).strip().lower(), np.random.randn(self.get_embedding_dimension())
        )
        return vector

    def get_tokens(self, input_values: Iterable[str]) -> Set[str]:
        """
        Extract the most representative tokens of each value and return the token set.
        Here, the most representative tokens are the ones with the lowest TF/IDF scores -
        tokens that describe what the values are about.
        Parameters
        ----------
        input_values : Iterable[str]
            The collection of values to extract tokens from.

        Returns
        -------
        Set[str]
            A set of representative tokens
        """

        if len(input_values) < 1:
            return set()

        try:
            vectorizer = TfidfVectorizer(
                decode_error="ignore",
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                stop_words=self._stop_words,
                token_pattern=self._token_pattern,
                max_df=self._max_df,
                use_idf=True,
            )
            vectorizer.fit_transform(input_values)
        except ValueError:
            return set()

        weight_map = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tokenset = set()
        tokenizer = vectorizer.build_tokenizer()
        for value in input_values:
            value = value.lower().replace("\n", " ").strip()
            for shingle in shingles(value):
                tokens = [t for t in tokenizer(shingle)]

                if len(tokens) < 1:
                    continue

                token_weights = [weight_map.get(t, 0.0) for t in tokens]
                min_tok_id = np.argmin(token_weights)
                tokenset.add(tokens[min_tok_id])

        return tokenset

    def transform(self, input_values: Iterable[str]) -> np.ndarray:
        """
         Extract the embeddings of the most representative tokens of each value and return their **mean** embedding.
         Here, the most representative tokens are the ones with the lowest TF/IDF scores -
         tokens that describe what the values are about.
         Given that the underlying embedding model is a n-gram based one,
         the number of out-of-vocabulary tokens should be relatively small or zero.
         Parameters
         ----------
        input_values : Iterable[str]
             The collection of values to extract tokens from.

         Returns
         -------
         np.ndarray
             A Numpy vector representing the mean of all token embeddings.
        """

        embeddings = [self.get_vector(token) for token in self.get_tokens(input_values)]
        if len(embeddings) == 0:
            return np.empty(0)
        return np.mean(np.array(embeddings), axis=0)


def compute_value_signature(col_list, feature_extractor, num_val_samples):
    # randomly pick a corresponding table column to compute value signature
    count = 0

    while count < VALUE_SIGNATURE_ATTEMPTS:
        rnd_idx = random.randrange(len(col_list))
        table_id, col_name = col_list[rnd_idx]
        df = readCSVFileWithTableID(table_id, usecols=[col_name], nrows=num_val_samples).astype(str)
        fasttext_signature = feature_extractor.transform(df[col_name].tolist())
        if len(fasttext_signature) != 0:
            break
        else:
            count += 1

    return fasttext_signature


def compute_name_similarity(name_sig1, name_sig2):
    return len(name_sig1.intersection(name_sig2)) / len(name_sig1.union(name_sig2))


def compute_value_similarity(value_sig1, value_sig2):
    return np.dot(value_sig1, value_sig2) / (np.linalg.norm(value_sig1) * np.linalg.norm(value_sig2) + CONSTANT)
