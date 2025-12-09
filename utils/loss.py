import torch.nn as nn
import torch

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.use_phi = loss_params.get('use_phi')

    def get_ones_V_P(self, batch_size, device):
        """
        引入谓词,构造V和P矩阵
        问题：p矩阵和V矩阵是否需要归一化？
        """
        # phi=1
        batch_size = batch_x.shape[0]
        V = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        # P = torch.ones((batch_size, batch_size), device=device)  # 全1矩阵
        phi = batch_x[:, :, 1].mean(dim=1)     
        phi = phi / torch.linalg.norm(phi)
        phi = phi.unsqueeze(0)       

        P = torch.mm(phi, phi.T)
        return V, P

    def get_x_V_P(self, batch_x, batch_y, batch_size, c_norms, device):
        """
        引入谓词,构造V和P矩阵
        问题：p矩阵和V矩阵是否需要归一化？
        """
        batch_size = batch_x.shape[0]
        V = torch.eye(batch_size, device=device)  # 单位矩阵 (batch_size x batch_size)
        #P = torch.zeros((batch_size, batch_size), device=device)
        #phi = c_norms # b s
        #phi = phi / torch.linalg.norm(phi)
        #P = torch.mm(phi,phi.T) # bxb

        #ret均值
        phi = batch_x[:,:,1].mean(dim=1)
        phi = (phi / torch.linalg.norm(phi)).unsqueeze(0)
        P = torch.mm( phi.T,phi)    

        return V, P
        # for m in range(len(c_norms)):
        #     temp_phi = batch_x[:, -1, m].unsqueeze(0) # phixq
        # #     temp_phi = batch_x[:, :, m].mean(dim=1).unsqueeze(0) # phixm
        #     temp_phi = (temp_phi / c_norms[m]).unsqueeze(0) # 全局归一化
        #     temp_phi = temp_phi / torch.linalg.norm(temp_phi, ord=2) # 在bs内在做一次归一化将值放大，全局的时候用
        #     temp_P = torch.mm(temp_phi.T, temp_phi) # bxb
        #     P = P + temp_P

        # phi=x(所有特征)
        # phi = batch_x[:,-1,:]
        # # phi = phi * phi.T # phi=xx^T
        # phi = phi / torch.linalg.norm(phi)
        # P = torch.mm(phi,phi.T) # bxb

        # phi=xx^T
        # _, _, C = batch_x.shape
        # phi = batch_x[:,-1,:].view(batch_size, 1, C)
        # phi = torch.bmm(phi, phi.transpose(-2, -1)).squeeze(-1).squeeze(-1)
        # phi = phi / torch.linalg.norm(phi)
        # phi = phi.unsqueeze(-1)
        # P = torch.mm(phi,phi.T) # bxb

        # phi=f
        # phi = c_norms[:,-1,:]
        # phi = phi / torch.linalg.norm(phi)
        # P = torch.mm(phi,phi.T) # bxb

        # phi=y
        # phi = batch_y.unsqueeze(-1) # b
        # phi = phi / torch.linalg.norm(phi)
        # P = torch.mm(phi,phi.T) # bxb

        # phi=y history
        

    def forward(self, batch_x, outputs, targets, tau_hat=None, tau=None, c_norms=None):
        batch_size = outputs.shape[0]
        device = outputs.device
        # V, P = self.get_ones_V_P(batch_size, device) # phi=1

        # V_ones, P_ones = self.get_ones_V_P(batch_size, device) # phiplus1
        V, P = self.get_x_V_P(batch_x, targets, batch_size, c_norms, device) # phi=other
        # P = P + P_ones
        error = outputs - targets
        if tau_hat is None or tau is None:
            weighted_error = error
            return {
                # 'total':  torch.nn.functional.mse_loss(outputs, targets),
                'total':  torch.mean(weighted_error ** 2),
                'mse': torch.mean(weighted_error ** 2),
                'V_loss': torch.mean(torch.matmul(V, error)** 2),
                'P_loss': torch.mean(torch.matmul(P, error)** 2),
            }
        else:
            weight_matrix = tau_hat * V + tau * P # V保证自我, P batch内特征相似度 相似度越高对应P里面位置的数越大，如果这时候误差很大就会放大误差做惩罚
            weighted_error = torch.matmul(weight_matrix, error)
            return {
                'total': torch.mean(weighted_error ** 2),
                'mse': torch.mean(weighted_error ** 2),
                'V_loss': torch.mean(torch.matmul(tau_hat * V, error)** 2),
                'P_loss': torch.mean(torch.matmul(tau * P, error)** 2), # 特征相似度的误差
                }