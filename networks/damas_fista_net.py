import torch 
import torch.nn as nn


class DAMAS_FISTANet(nn.Module):
    def __init__(self, LayerNo, wk_reshape, A,  show_DAS_result=False):
        super(DAMAS_FISTANet, self).__init__()

        # learnable parameter
        self.lambda_step = nn.Parameter(torch.Tensor([1 for i in range(LayerNo)]))
        self.y_step = nn.Parameter(torch.Tensor([1 for i in range(LayerNo)]))
        self.LayerNo = LayerNo
        self.show_DAS_result = show_DAS_result
     
        # wk_reshape.size = 1×56×1681
        self.wk_reshape = wk_reshape.reshape(1, wk_reshape.shape[0], wk_reshape.shape[1])
        self.wk_weight = nn.Parameter(torch.Tensor([1]))
        # A.size = 1×1681×1681
        self.A = A.reshape(1, A.shape[0], A.shape[1])

        # thresholding value
        self.L = torch.Tensor([3250]).cuda()

        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([1]))
        self.b_rho = nn.Parameter(torch.Tensor([1]))
        self.Sp = nn.Softplus()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1681, 1681)
        self.weight1 = nn.Parameter(torch.Tensor([1]))
        self.weight2 = nn.Parameter(torch.Tensor([1]))


    def forward(self, CSM):
        
        # 麦克风个数和频点个数
        # CSM_size: batch×56×56×1
        N_mic = CSM.shape[1]


        # 频率 K 下的波束成像图
        # 取最后一维——单频点
        # CSM_k_size：batch×56×56
        # wk_reshape.size = 1×56×1681
        CSM_k = CSM[:, :, :, 0]
        # DAS_results.size = batch×1×1681
        DAS_results = torch.sum(torch.mul(self.wk_weight*self.wk_reshape.conj(), torch.matmul(CSM_k, self.wk_weight*self.wk_reshape)), dim=1, keepdim=True) / (N_mic ** 2)
        # DAS_results.size = batch×1681×1
        DAS_results = torch.real(DAS_results).permute(0, 2, 1)

        if self.show_DAS_result:
            return DAS_results

        # 迭代过程    
        xold = torch.zeros(DAS_results.shape).to(torch.float64).cuda()
        y = xold 

        # 参数准备
        # ATA.size = 1×1681×1681
        ATA = torch.matmul(self.A.permute(0, 2, 1), self.A)
        # DAS_results.size = batch×1681×1
        ATb = torch.matmul(self.A.permute(0, 2, 1), DAS_results)

        # FISTA-Net迭代部分
        for i in range(self.LayerNo):

            r_n = self.y_step[i] * y - (1 / self.L) * self.lambda_step[i] * torch.matmul(ATA, y) + (1 / self.L) * self.lambda_step[i] * ATb

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