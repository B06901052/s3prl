import torch

PATH = 'result/downstream/Exp003-2/states-200000.ckpt'
PATH2 = 'result/downstream/Exp003-2/pca'

checkpoint = torch.load(PATH)
pca = checkpoint['Feature_Transformer']

torch.save(pca,PATH2)

