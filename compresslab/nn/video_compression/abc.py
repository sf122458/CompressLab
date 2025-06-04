from abc import ABC

class PFrameCodec(ABC):
    """
    Abstract base class for P-frame codecs in video compression.
    This class inherits from `CompressionModel` and serves as a template for
    implementing specific P-frame codecs.
    """

class IPFrameCodec(ABC):
    """
    Abstract base class for I-frame and P-frame codecs in video compression.
    This class inherits from `CompressionModel` and serves as a template for
    implementing specific I-frame codecs.
    """
