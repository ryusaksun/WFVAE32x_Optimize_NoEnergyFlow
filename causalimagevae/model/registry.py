class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, model_name):
        def decorator(model_class):
            cls._models[model_name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._models:
            raise KeyError(
                f"Model '{model_name}' is not registered. "
                f"Available models: {list(cls._models.keys())}"
            )
        return cls._models[model_name]