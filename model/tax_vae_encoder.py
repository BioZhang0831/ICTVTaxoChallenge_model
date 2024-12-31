import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
#  Encoder：TaxonEncoder
# -----------------------------
class TaxonEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, latent_dim):
        super(TaxonEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        #self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_units, batch_first=True)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_units,       
            num_layers=5,          
            batch_first=True,
            dropout=0.3           
        )
        # 输出均值和对数方差
        self.z_mean = nn.Linear(lstm_units, latent_dim)
        self.z_log_var = nn.Linear(lstm_units, latent_dim)

    def forward(self, x):
        """
        x.shape = [batch_size, max_seq_len]
        """
        # 1) 嵌入
        embedded = self.embedding(x)   #
        _, (h_n, _) = self.lstm(embedded)  
        lstm_out = h_n[-1]      
        z_mean = self.z_mean(lstm_out)     
        z_log_var = self.z_log_var(lstm_out)
        return z_mean, z_log_var

# -----------------------------
#  Decoder：TaxonDecoder
# -----------------------------
class TaxonDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, latent_dim):
        super(TaxonDecoder, self).__init__()
        # 将 z 向量沿序列维度 repeat
        self.lstm_units = lstm_units
        self.latent_dim = latent_dim
        self.repeat_vector = nn.Linear(latent_dim, embed_dim)
        # 解码 LSTM
        #self.lstm = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, batch_first=True)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_units,       
            num_layers=5,          
            batch_first=True,
            dropout=0.3           
        )
        # 输出层：将 LSTM hidden -> vocab 概率分布
        self.output_layer = nn.Linear(lstm_units, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, z):
        # [batch_size, latent_dim] -> [batch_size, 128]
        h_init = self.repeat_vector(z)

        # repeat到 seq_len 维度: [batch_size, max_seq_len, 128]
        max_seq_len = 6  # 如果是固定为 6
        h_init = h_init.unsqueeze(1).repeat(1, max_seq_len, 1)

        # lstm_out: [batch_size, max_seq_len, 128]
        lstm_out, _ = self.lstm(h_init)

        # 输出层 + log_softmax
        logits = self.output_layer(lstm_out)
        #log_probs = self.softmax(logits)  # [batch_size, max_seq_len, vocab_size]
        return logits

# -----------------------------
#  VAE：TaxonVAE（组合Encoder+Decoder）
# -----------------------------
class TaxonVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_units, latent_dim):
        super(TaxonVAE, self).__init__()
        self.encoder = TaxonEncoder(vocab_size, embed_dim, lstm_units, latent_dim)
        self.decoder = TaxonDecoder(vocab_size, embed_dim, lstm_units, latent_dim)

    def reparameterize(self, z_mean, z_log_var):
        """
        Reparameterization Trick:
          z = μ + ε * σ
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        """
          x.shape = [batch_size, max_seq_len]
        """
        # 编码阶段
        z_mean, z_log_var = self.encoder(x)
        # 采样
        z_sample = self.reparameterize(z_mean, z_log_var)
        # 解码阶段
        log_probs = self.decoder(z_sample)
        return log_probs, z_mean, z_log_var

def vae_loss_fn(x, logits, z_mean, z_log_var):
    # Use CrossEntropyLoss directly
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    # Reshape logits to [batch_size * seq_len, vocab_size]
    logits = logits.view(-1, logits.size(-1))
    # Flatten targets to [batch_size * seq_len]
    targets = x.view(-1)
    recon_loss = loss_fn(logits, targets)
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)
    
    return recon_loss + kl_loss, recon_loss, kl_loss



node_name_to_id = {}
current_id = 1
class_label_dict = {}

def get_node_id(name):
    global current_id
    if name not in node_name_to_id:
        node_name_to_id[name] = current_id
        current_id += 1
    return node_name_to_id[name]


import pickle
with open('tax/tmp/clas_dict.pickle', 'rb') as f:
    clas_dict = pickle.load(f)


for key in clas_dict.keys():
    hc = []
    for i in range(6):
        if clas_dict[key][i] is not None:
            hc.append(get_node_id(clas_dict[key][i]))
        else:
            hc.append(0)
    class_label_dict[key] = hc


max_seq_len = 6           # taxonomic lineage 的长度 (phylum->class->order->family->genus->species)
vocab_size=len(node_name_to_id)+1
embed_dim = 256            # Embedding层输出维度
lstm_units = 256           # LSTM隐藏单元数
latent_dim = 64           # VAE潜在空间维度
batch_size = 16           # 训练批次大小

x_train=torch.tensor(list(class_label_dict.values()))
model = TaxonVAE(vocab_size, embed_dim, lstm_units, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
num_epochs = 20

for epoch in range(num_epochs):
    optimizer.zero_grad()
    log_probs, z_mean, z_log_var = model(x_train)
    
    # 计算 loss
    loss, recon_loss, kl_loss = vae_loss_fn(x_train, log_probs, z_mean, z_log_var)
    
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.4f} | "
        f"Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")

model.eval()
with torch.no_grad():
    # 1) 编码
    z_mean, z_log_var = model.encoder(x_train)
    # 2) 采样
    z_sample = model.reparameterize(z_mean, z_log_var)
    # 3) 解码：得到 log 概率，再取 argmax 作为预测
    log_probs = model.decoder(z_sample)  # [batch_size, max_seq_len, vocab_size]
    preds = torch.argmax(log_probs, dim=-1)  # [batch_size, max_seq_len]

# 输出前几个样本的原始输入与解码出的结果以对比
#print("Example original input (batch[0]):", x_train[0])
#print("Example reconstruction prediction (batch[0]):", preds[0])


# 保存模型
torch.save(model, 'taxon_vae.pth')
print("Model saved!")