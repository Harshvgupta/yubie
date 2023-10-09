from enum import Enum

# Enumerators
Topics = Enum('Topics', [
    'ImageFromBot',
    'AudioFromUser',
    'CommandFromUser',
    'DetectionsFromVision'
    # Has to be extended
])
