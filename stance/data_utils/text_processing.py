import string


def remove_non_printable(text):
    return ''.join(s for s in text
                   if s in string.printable)


def normalize_punct(text, lang):
    """ Normalize punctuation of a given string
    Punctuation normalizer:
    https://bitbucket.org/luismsgomes/mosestokenizer
    """
    from mosestokenizer import MosesPunctuationNormalizer
    with MosesPunctuationNormalizer(lang) as normalize:
        return normalize(text)


def string_descape(text):
    """ Descape a string with escaped characters """
    return text.decode('string_escape')


def string_unescape(s):
    return (s.replace("&lt;", "<")
             .replace("&apos;", "'")
             .replace("&gt;", ">")
             .replace("&quot;", "\"")
             .replace("&amp;", "&")
             .replace("@-@", "-"))


def _moses_tokenize(text, lang):
    """ Tokenize a given string using moses tokenizer
    Tokenization: https://github.com/alvations/sacremoses
    """
    from sacremoses import MosesTokenizer
    mt = MosesTokenizer(lang)
    return [string_unescape(t)
            for t in mt.tokenize(text)]


def tokenize(text, lang='en', lower_case=True, descape=False):
    """Tokenize an input string and return the preprocessed output string.
    Instead of using the external script using python libraries.
    """
    # normalize punct and remove non printable characters
    text = normalize_punct(text, lang)

    # remove non printable chars
    text = remove_non_printable(text)

    if descape:
        text = string_descape(text)

    # moses tokenizer
    text = " ".join(_moses_tokenize(text, lang))

    return text
