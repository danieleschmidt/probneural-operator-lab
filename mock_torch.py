"""Mock PyTorch for testing syntax validation without full torch installation."""

class MockTensor:
    def __init__(self, *args, **kwargs):
        pass
    
    def to(self, *args, **kwargs):
        return self
    
    def cpu(self):
        return self
    
    def detach(self):
        return self
    
    def numpy(self):
        return None
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
    
    def __call__(self, *args, **kwargs):
        return self

class MockModule:
    def __init__(self, *args, **kwargs):
        pass
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []
    
    def train(self, mode=True):
        return self
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def forward(self, *args, **kwargs):
        return MockTensor()
    
    def __call__(self, *args, **kwargs):
        return MockTensor()
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: MockTensor()

class MockOptim:
    class Adam:
        def __init__(self, *args, **kwargs):
            pass
        def step(self):
            pass
        def zero_grad(self):
            pass

class MockNN:
    Module = MockModule
    
    class Linear(MockModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = MockTensor()
            self.bias = MockTensor()
    
    class Conv2d(MockModule):
        pass
    
    class Dropout(MockModule):
        pass
    
    class MSELoss(MockModule):
        pass

def tensor(*args, **kwargs):
    return MockTensor()

def zeros(*args, **kwargs):
    return MockTensor()

def ones(*args, **kwargs):
    return MockTensor()

def randn(*args, **kwargs):
    return MockTensor()

def stack(*args, **kwargs):
    return MockTensor()

def cat(*args, **kwargs):
    return MockTensor()

device = "cpu"

nn = MockNN()
optim = MockOptim()

class autograd:
    class Function:
        @staticmethod
        def forward(ctx, *args):
            return MockTensor()
        
        @staticmethod
        def backward(ctx, *args):
            return MockTensor()
    
    @staticmethod
    def grad(*args, **kwargs):
        return [MockTensor()]

fft = type('fft', (), {
    'fft': lambda x: MockTensor(),
    'ifft': lambda x: MockTensor(),
    'rfft': lambda x: MockTensor(),
    'irfft': lambda x: MockTensor(),
})()