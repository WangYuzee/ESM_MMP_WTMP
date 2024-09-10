import torch
import esm
import tqdm
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Process some filePath.')
parser.add_argument('-i', type=str, default="")

parser.add_argument('-o',type=str,default="./")
args = parser.parse_args()
inputfile=args.i

out_path=args.o

inputname2=inputfile.split('/')[-1].strip('.pt')


#designed_seq=json.load(open(inputfile,'r'))
with open(inputfile) as f:
    designed_seq=f.readlines()

# 设置新的模型缓存位置
os.environ['TORCH_HOME'] = './Model'
model_name = 'esm2_t33_650M_UR50D'
model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)

residue_codebook = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']



def WildtypeMP(model,alphabet,S_wt,S_mut):
    assert len(S_wt) == len(S_mut)
    mask_token = alphabet.get_tok(alphabet.mask_idx)
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("", S_wt)])
    tokens = tokens.to('cuda')
    model = model.to('cuda')
    with torch.no_grad():
        result = model(tokens,return_contacts=False)
    probabilties = result['logits'][0, [i+1 for i in range(len(S_wt))], 4:24].softmax(-1) 
    score = 0
    for position in range(len(S_wt)):
        score += torch.log(probabilties[position,residue_codebook.index(S_mut[position])]) - torch.log(probabilties[position,residue_codebook.index(S_wt[position])])
    return float(score)

def MaskMP_A(model, alphabet, S_wt, S_mut):
    assert len(S_wt) == len(S_mut)
    mask_token = alphabet.get_tok(alphabet.mask_idx)
    converter = alphabet.get_batch_converter()
    masked_sequence = ''.join([mask_token for _ in list(S_wt)])
    _, _, mask_tokens = converter([("", masked_sequence)])
    mask_tokens = mask_tokens.to('cuda')
    model = model.to('cuda')
    mask_tokens.to('cuda')
    with torch.no_grad():
        result = model(mask_tokens,return_contacts=False) # result['logits']num_sequences, num_residues+2, 33
    probabilties = result['logits'][0, [i+1 for i in range(len(S_wt))], :].softmax(-1) # [num_residues,33]
    score = 0
    for position in range(len(S_wt)):
        score += torch.log(probabilties[position,residue_codebook.index(S_mut[position])]) - torch.log(probabilties[position,residue_codebook.index(S_wt[position])])
    return float(score)

#test MaskMP_A

# MaskMP_A(model=model,alphabet = alphabet, S_wt=S_wt,S_mut=S_mut)
# WildtypeMP(model=model,alphabet = alphabet, S_wt=S_wt,S_mut=S_mut)


# 创建热图数组

# 确保 designed_seq 是一个字符串列表
designed_seq = list(designed_seq[1].strip())

heatmap_MMP = np.zeros((len(residue_codebook), len(designed_seq)))
heatmap_WTMP = np.zeros((len(residue_codebook), len(designed_seq)))
print(f"Shape of heatmap_MMP: {heatmap_MMP.shape}")
print(f"Shape of heatmap_WTMP: {heatmap_WTMP.shape}")

result_dict_MMP = {}
result_dict_WTMP = {}
for i in tqdm.tqdm(range(len(designed_seq))):
    for j in range(len(residue_codebook)):
        # 使用列表复制以保留原始序列
        S_wt = designed_seq.copy()
        S_mut = S_wt.copy()
        
        # 修改 S_mut 中的字符
        S_mut[i] = residue_codebook[j]
        S_mut = ''.join(S_mut)
        S_wt = ''.join(S_wt)

        # 计算 MMP 和 WTMP 值
        MMP = MaskMP_A(model=model, alphabet=alphabet, S_wt=S_wt, S_mut=S_mut)
        heatmap_MMP[j, i] = MMP

        WTMP = WildtypeMP(model=model, alphabet=alphabet, S_wt=S_wt, S_mut=S_mut)
        heatmap_WTMP[j, i] = WTMP
        result_dict_MMP['>'+S_wt[i]+str(i+1)+S_mut[i]+" MMP score: "+str(MMP)] = [S_mut, MMP]
        result_dict_WTMP['>'+S_wt[i]+str(i+1)+S_mut[i]+" WTMP score: "+str(WTMP)] = [S_mut, WTMP]


#根据得分排序
result_dict_MMP = dict(sorted(result_dict_MMP.items(), key=lambda x: x[1][1], reverse=True))
result_dict_WTMP = dict(sorted(result_dict_WTMP.items(), key=lambda x: x[1][1], reverse=True))


#保存结果成fasta格式
with open(out_path+'MMP_result'+inputname2+'.fasta','w') as f:
    for key in result_dict_MMP:
        f.write(key+'\n')
        f.write(result_dict_MMP[key][0]+'\n')
with open(out_path+'WTMP_result'+inputname2+'.fasta','w') as f:
    for key in result_dict_WTMP:
        f.write(key+'\n')
        f.write(result_dict_WTMP[key][0]+'\n')








# 打印调试信息
print(f"Shape of heatmap_MMP: {heatmap_MMP.shape}")
print(f"Shape of heatmap_WTMP: {heatmap_WTMP.shape}")

#绘制热图

plt.figure(figsize=(len(designed_seq)/8, 5))
plt.imshow(heatmap_MMP, cmap="viridis", aspect="auto")
plt.xticks(range(len(designed_seq) ), designed_seq)
plt.yticks(range(20), ''.join(residue_codebook))
plt.xlabel("Position in Protein Sequence")
plt.ylabel("Amino Acid Mutations")
plt.title("Predicted Effects of Mutations on Protein Sequence (MMP)")
plt.colorbar(label="Log Likelihood Ratio (LLR)")
plt.show()
plt.savefig(out_path+'MMP_heatmap'+inputname2+'.pdf')
#保存成pdf
plt.figure(figsize=(len(designed_seq)/8, 5))
plt.imshow(heatmap_WTMP, cmap="viridis", aspect="auto")
plt.xticks(range(len(designed_seq) ), designed_seq)
plt.yticks(range(20), ''.join(residue_codebook))
plt.xlabel("Position in Protein Sequence")
plt.ylabel("Amino Acid Mutations")
plt.title("Predicted Effects of Mutations on Protein Sequence (WTMP)")
plt.colorbar(label="Log Likelihood Ratio (LLR)")
plt.show()
plt.savefig(out_path+'WTMP_heatmap'+inputname2+'.pdf')


