import torch.cuda
import tqdm

print(torch.cuda.is_available())

#for i in tqdm.tqdm(np.arange(1000)):

torch.randint(low=0, high=64, size=(10,128))
