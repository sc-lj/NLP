# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals


from functools import wraps

def cached_property(getter):
    """
    Decorator that converts a method into memoized property.
    The decorator works as expected only for classes with
    attribute '__dict__' and immutable properties.
    """
    @wraps(getter)
    def decorator(self):
        key = "_cached_property_" + getter.__name__

        if not hasattr(self, key):
            setattr(self, key, getter(self))

        return getattr(self, key)

    return property(decorator)

def to_unicode(object):
    if isinstance(object,str):
        return object
    elif isinstance(object,bytes):
        return object.decode("utf-8")
    else:
        if hasattr(object, "__str__"):
            return str(object)
        elif hasattr(object, "__bytes__"):
            return bytes(object).decode("utf-8")

def unicode_compatible(cls):
    """
    用于unicode兼容类的装饰器，将方法__unicode__应用装饰器中。
    :param cls:
    :return:
    """
    cls.__str__=cls.__unicode__
    cls.__bytes__=lambda self:self.__str__().encode("utf-8")
    return cls

@unicode_compatible
class Sentence(object):
    __slots__ = ("_text", "_cached_property_words", "_tokenizer", "_is_heading",)

    def __init__(self, text, is_heading=False):
        self._text = to_unicode(text).strip()
        self._is_heading = bool(is_heading)


    @property
    def is_heading(self):
        return self._is_heading

    def __eq__(self, sentence):
        assert isinstance(sentence, Sentence)
        return self._is_heading is sentence._is_heading and self._text == sentence._text

    def __ne__(self, sentence):
        return not self.__eq__(sentence)

    def __hash__(self):
        return hash((self._is_heading, self._text))

    def __unicode__(self):
        return self._text

    def __repr__(self):
        return to_unicode("<%s: %s>") % (
            "Heading" if self._is_heading else "Sentence",
            self.__str__()
        )
