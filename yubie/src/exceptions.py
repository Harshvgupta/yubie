class TopicNotRegisteredException(Exception):
    def __init__(self, topic):
        return f'{topic} is not registered. Register the topic in main.py file'


class ModelNotFoundException(Exception):
    def __init__(self, model):
        return f'{model} is not registered. \
        Check whether the model {model} is registered using `registerModel` method'
