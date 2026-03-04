class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                shadow = self.shadow[name]
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * shadow
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                shadow = self.shadow[name]
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                    self.shadow[name] = shadow
                param.data = shadow.clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data = self.backup[name]
        self.backup = {}
        
        