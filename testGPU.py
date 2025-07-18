import torch

print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
