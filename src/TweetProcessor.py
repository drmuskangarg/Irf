from typing import Iterable
from emoji import demojize, is_emoji
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import contractions
import re
import spacy
from spacy.tokenizer import _get_regex_pattern


class TweetProcessor:
    def __init__(self, lowercase=True, remove_stopwords=False, lemmatize=False,
                 replace_urls='', replace_mentions='', replace_hashtags=False,
                 convert_to_ascii=False, remove_punctuation=True,
                 remove_numbers=True, replace_emojis=True,
                 expand_contractions=True) -> None:

        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.replace_urls = replace_urls
        self.replace_mentions = replace_mentions
        self.replace_hashtags = replace_hashtags
        self.convert_to_ascii = convert_to_ascii
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.replace_emojis = replace_emojis
        self.expand_contractions = expand_contractions

        self.__tokenizer__ = TweetTokenizer()

        if remove_stopwords:
            # nltk.download('stopwords')
            self.__stopwords__ = set(stopwords.words('english'))
        
        if lemmatize:
            self.__spacy_en__ = spacy.load('en_core_web_sm')
            # get default pattern for tokens that don't get split
            re_token_match = _get_regex_pattern(self.__spacy_en__.Defaults.token_match)
            # add your patterns (here: hashtags and in-word hyphens)
            re_token_match = f"({re_token_match}|#\w+|\w+-\w+)"

            # overwrite token_match function of the tokenizer
            self.__spacy_en__.tokenizer.token_match = re.compile(re_token_match).match


    def transform(self, text: str) -> str:
        if self.expand_contractions:
            text = contractions.fix(text)
        
        tokens = self.tokenize(text)
        text = " ".join([self.__normalizeToken__(token) for token in tokens])

        if self.lowercase:
            text = text.lower()

        if self.lemmatize:
            text = self.__lemmatize__(text)

        if self.convert_to_ascii:
            text = text.encode('ascii', 'ignore').decode()
        
        text = re.sub('\s+', ' ', text)

        return text

    def tokenize(self, *args, **kwargs) -> Iterable:
        return self.__tokenizer__.tokenize(*args, **kwargs)
        

    def __normalizeToken__(self, token):
        lowercased_token = token.lower()
        if token.startswith("@") and self.replace_mentions != False:
            return self.replace_mentions
        
        elif (lowercased_token.startswith("http")
                or lowercased_token.startswith("www")
                and self.replace_urls != False):
            return self.replace_urls
        
        elif is_emoji(token) and self.replace_emojis != False:
            if type(self.replace_emojis) == str:
                return self.replace_emojis
            return demojize(token)
        
        elif token.startswith('#') and len(token) > 1 and self.replace_hashtags != False:
            return self.replace_hashtags

        elif token == "’":
            return "'"
        elif token == "…":
            return "..."

        elif len(token) == 1 and self.remove_punctuation:
            return token.strip("!”#$%&()*+,-./:;<=>?@[\]^_`{|}~")

        elif token.isnumeric() and self.remove_numbers:
            return ''

        elif self.remove_stopwords and lowercased_token in self.__stopwords__:
            return ''

        else:
            return token

    def __lemmatize__(self, text):
        return ' '.join([t.lemma_ for t in self.__spacy_en__(text)])
