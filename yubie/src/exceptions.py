class TopicNotRegisteredException(Exception):
    def __init__(self, topic):
        message = f'{topic} is not registered. Register the topic in main.py file'
        super().__init__(message)


class ModelNotFoundException(Exception):
    def __init__(self, model):
        message = f'Model Name `{model}` is not registered. Check whether the model {model} is registered using `registerModel` method'
        super().__init__(message)
