import torch 
import torch.nn as nn


class DAMAS_FISTANet(nn.Module):
    def __init__(self, LayerNo, show_DAS_result=False):
        super(DAMAS_FISTANet, self).__init__()

        # learnable parameter
        self.lambda_step = nn.Parameter(torch.Tensor([1 for i in range(LayerNo)]))
        self.y_step = nn.Parameter(torch.Tensor([1 for i in range(LayerNo)]))
        self.LayerNo = LayerNo
        self.show_DAS_result = show_DAS_result
     
        self.wk_weight = nn.Parameter(torch.Tensor([1]))

        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([1]))
        self.b_rho = nn.Parameter(torch.Tensor([1]))
        self.Sp = nn.Softplus()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1681, 1681)
        self.weight1 = nn.Parameter(torch.Tensor([1]))
        self.weight2 = nn.Parameter(torch.Tensor([1]))


    def forward(self, CSM_K, wk_reshape_K, A_K, ATA_K, L_K):
        
        # 麦克风个数和频点个数
        # CSM_size: batch×56×56×1
        N_mic = CSM_K.shape[1]


        # 频率 K 下的波束成像图
        DAS_results_K = torch.sum(torch.mul(self.wk_weight*wk_reshape_K.conj(), torch.matmul(CSM_K, self.wk_weight*wk_reshape_K)), dim=1, keepdim=True) / (N_mic ** 2)
        DAS_results_K = torch.real(DAS_results_K).permute(0, 2, 1)


        if self.show_DAS_result:
            return DAS_results_K

        # 迭代过程    
        xold = torch.zeros(DAS_results_K.shape).to(torch.float64).cuda()
        y = xold 

        # 参数准备
        # ATA.size = 1×1681×1681
        # DAS_results.size = batch×1681×1
        ATb_K = torch.matmul(A_K.permute(0, 2, 1), DAS_results_K)

        # FISTA-Net迭代部分
        for i in range(self.LayerNo):

            r_n = self.y_step[i] * y - (1 / L_K) * self.lambda_step[i] * torch.matmul(ATA_K, y) + (1 / L_K) * self.lambda_step[i] * ATb_K
            xnew = self.relu(r_n)
            y = self.w_rho*xnew +  self.b_rho * (xnew - xold)
            xold = xnew
        
        # 加权全连接层
        temp = xnew
        xnew = torch.squeeze(xnew, 2).to(torch.float32)
        xnew = self.linear(xnew)        
        xnew = torch.unsqueeze(xnew, 2).to(torch.float64)
        xnew = self.relu(self.weight1*xnew + self.weight2*temp)
    
        return xnew