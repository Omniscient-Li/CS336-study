import torch
import torch.nn as nn
import torch.nn.functional as F

class RopelessMLA(nn.Module):
    def __init__(self, d_model , n_heads , kv_latent_dim):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads # dimension of each head

        #projection layers
        self.W_q = nn.Linear(d_model , d_model , bias = False) #Query projection
        self.W_dkv = nn.Linear(d_model , kv_latent_dim , bias = False) # Compress into latent KV space
        self.W_uk = nn.Linear(kv_latent_dim , d_model , bias = False) # Decompress K
        self.W_uv = nn.Linear(kv_latent_dim , d_model , bias = False) # Decompress V
        self.W_o = nn.Linear(d_model , d_model , bias = False) # Final output projection

        self.kv_latent_dim = kv_latent_dim
        self.ln = nn.LayerNorm(kv_latent_dim)

    def absorbed_k(self):
        H , dh , D , latent = self.n_heads , self.dh , self.d_model , self.kv_latent_dim
        Wq_heads = self.W_q.weight.view(H , dh , D) # (H , dh , D)
        Wuk_heads = self.W_uk.weight.view(H , dh , latent) # (H , dh , latent)
        return torch.einsum('hdi,hdl->hil' , Wq_heads , Wuk_heads) # (H , D , latent)

    def forward(self , x , kv_cache = None , past_length = 0):
        B , S , D = x.size()

        absorbed_k = self.absorbed_k()

        #Compress X into latent KV space
        new_c_kv = self.ln(self.W_dkv(x)) # (B , S , latent_dim)
        if kv_cache is None:
            c_kv = new_c_kv
        else:
            c_kv = torch.cat([kv_cache , new_c_kv] , dim = 1) # (B , S_total , latent_dim)

        S_full = c_kv.size(1)

        #Compress V to full d_model and split into heads
        v_full = self.W_uv(c_kv) # shape : (B , S_full , D)
        v = v_full.view(B , S_full , self.n_heads , self.dh).transpose(1 , 2) # shape : (B , n_heads , S_full , dh)

        # Compute attention scores
        attn_scores = torch.zeros(B , self.n_heads , S , S_full , device = x.device , dtype = x.dtype)
        for h in range(self.n_heads):
            tmp = torch.matmul(x , absorbed_k[h]) # shape : (B , S , latent_dim)
            attn_scores[: , h] = torch.bmm(tmp , c_kv.transpose(1 , 2)) # shape : (B , S , S_full)

        #Scale and apply casual mask
        attn_scores = attn_scores / (self.dh ** 0.5)
        mask = torch.tril(torch.ones((S , S_full) , device = x.device) , diagonal = past_length)
        attn_scores = attn_scores.masked_fill(mask.view(1 , 1 , S , S_full) == 0 , float('-inf'))

        #softmax to get attention weights
        atten_weights = F.softmax(attn_scores , dim = -1) # shape : (B , n_heads , S , S_full)

        # Apply attention weights to each heads' V separately
        out_heads = []
        for h in range(self.n_heads):
            context_h = torch.matmul(atten_weights[: , h] , v[: , h]) # shape : (B , S , dh)
            out_heads.append(context_h)

        out = torch.cat(out_heads , dim = -1) # shape : (B , S , D)

        return self.W_o(out) , c_kv # Return output and updated kv_cache

# ============= demo for testing ===============

def demo():
    model = RopelessMLA(d_model = 512 , n_heads = 8 , kv_latent_dim = 256)
    x = torch.randn(1 , 5 , 512) # Batch = 1 , Seq_len = 5 , d_model = 512

    out , kv_cache = model(x)
    print(f"Output : {out.shape} , Cache : {kv_cache.shape}")

    #Memory comparison
    std_size = 2 * 2 * 10 * 512 * 4 / 1024 # KB (standard KV : B * 2 (K , V) * T * D * float32)
    latent_size = 1 * 2 * 10 * 256 * 4 / 1024 # KB (latent KV : B * T * latent_dim * float32)
    print(f"Memory :Standard = {std_size : .1f}KB , Latent = {latent_size : .1f}KB , Reduction = {std_size / latent_size : .1f}x")

def demo_cache_usage():
    torch.manual_seed(0)

    model = RopelessMLA(d_model = 8 , n_heads = 2 , kv_latent_dim = 4)

    #-------Step 1 : Initial input (sequence of 5 tokens)---------
    x1 = torch.randn(1 , 5 , 8) # Batch = 1 , Seq_len = 5 , d_model = 8
    out1 , cache1 = model(x1)

    print("Step 1 : Initial input")
    print(f"Output shape : {out1.shape}")
    print(f"Cache shape : {cache1.shape}")

    #-------Step 2 : New input (sequence of 3 tokens)---------
    x2 = torch.randn(1 , 1 , 8) # Batch = 1 , Seq_len = 1 , d_model = 8
    out2 , cache2 = model(x2 , kv_cache = cache1 , past_length = 5)

    print("\nStep 2 : New input with cache")
    print(f"Output shape : {out2.shape}")
    print(f"Cache shape : {cache2.shape}")

def demo_kv_cache_growth(num_initial_tokens = 5 , num_new_tokens = 3):
    torch.manual_seed(0)

    model = RopelessMLA(d_model = 8 , n_heads = 2 , kv_latent_dim = 4)

    # Step 1 : Start with initial tokens batch
    x = torch.randn(1 , num_initial_tokens , 8) # Batch = 1 , Seq_len = num_initial_tokens , d_model = 8
    out , cache = model(x)

    print(f"Step 0 : Intial input of {num_initial_tokens} tokens -> cache shape : {cache.shape}")

    # Step 2 : Incrementally append new tokens one at a time
    for step in range(1 , num_new_tokens + 1):
        new_token = torch.randn(1 , 1 , 8) # Batch = 1 , Seq_len = 1 , d_model = 8
        out , cache = model(new_token , kv_cache = cache , past_length = num_initial_tokens + step - 1)

        print(f"Step {step} : New token -> cache shape : {cache.shape}")




if __name__ == "__main__":
    demo()
    demo_cache_usage()
    demo_kv_cache_growth(num_initial_tokens = 50 , num_new_tokens = 4)

    