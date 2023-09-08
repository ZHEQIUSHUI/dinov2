import torch
logit = torch.rand([1,2,3,4])
bins = torch.rand([2])
output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)

print(output.shape)

output_matmul = (torch.matmul(torch.permute(logit,[0,2,3,1]), bins.unsqueeze(-1))).unsqueeze(dim=1).squeeze(dim=-1)
print(output_matmul.shape)
print(output - output_matmul)