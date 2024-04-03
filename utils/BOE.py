
import torch
import numpy as np

def calculate_ga_index(ga, size):
        # Map GA to the nearest increment starting from 20 (assuming a range of 20-40 GA)
        # size * (ga - min_ga) / (max_ga - min_ga)
        increment = (40-20)/size
        ga_mapped = torch.round((ga - 20) / increment)
        return ga_mapped   

def calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, α = 20, β = 40):
        # ( size / exp(a + b) ) * exp(a + ( b * (ga - min_ga) / (max_ga - min_ga) ) )
        ga_mapped = torch.round( (torch.tensor(size) / (torch.exp(torch.tensor(a + b)))) * 
                                (torch.exp(torch.tensor(a) + torch.tensor(b) * ((ga - α) /  (β - α)) )) )
        return ga_mapped  

def inv_calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, α = 20, β = 40):
        ga_mapped = torch.round( (torch.tensor(size) / (torch.exp(torch.tensor(a + b)))) * 
                                (torch.exp(torch.tensor(a) + torch.tensor(b) * ((-ga + β) /  (β - α)) )) )
        return ga_mapped  

def inv_inv_calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, α = 20, β = 40):
        ψ = calculate_ga_index_exp(40,size, a, b, α, β )
        ga_mapped = - inv_calculate_ga_index_exp(ga,size, a, b, α, β) + ψ
        return ga_mapped 

BOE_forms = {
            'BOE': calculate_ga_index,
            'EBOE': calculate_ga_index_exp,
            'inv_BOE': inv_calculate_ga_index_exp,
            'inv_inv_BOE': inv_inv_calculate_ga_index_exp
        }

def create_bi_partitioned_ordinal_vector(gas, size, BOE_form='BOE'):
        # Adjusting the threshold for the nearest 0.1 increment
        threshold_index = size//2
        device = gas.device
        batch_size = gas.size(0)
        # ga_indices = calculate_ga_index_exp(gas, size)
        ga_indices= BOE_forms[BOE_form](gas, size)
        vectors = torch.full((batch_size, size), -1, device=device)  # Default fill with -1

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > size:
                idx = size
            elif idx < 0:
                idx = 1
            
            if idx >= threshold_index:  # GA >= 30
                new_idx = (idx-threshold_index)*2
                vectors[i, :new_idx] = 1  # First 100 elements to 1 (up to GA == 30)
                vectors[i, new_idx:] = 0  # The rest to 0
            else:  # GA < 30
                new_idx = idx*2
                vectors[i, :new_idx] = 0  # First 100 elements to 0
                # The rest are already set to -1

        return vectors