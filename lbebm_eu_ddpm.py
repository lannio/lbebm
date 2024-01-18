import os
import math
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module, Linear

from torch.utils import data
from torch.autograd import Variable

import datetime, shutil, argparse, logging, sys

import utils
# add ddpm
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument("--data_scale", default=60, type=float)
    parser.add_argument("--dec_size", default=[1024, 512, 1024], type=list)
    parser.add_argument("--enc_dest_size", default=[256, 128], type=list)
    parser.add_argument("--enc_latent_size", default=[256, 512], type=list)
    parser.add_argument("--enc_past_size", default=[512, 256], type=list)
    parser.add_argument("--predictor_hidden_size", default=[1024, 512, 256], type=list)
    parser.add_argument("--non_local_theta_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_phi_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_g_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_dim", default=128, type=int)
    parser.add_argument("--fdim", default=16, type=int)
    parser.add_argument("--future_length", default=12, type=int)
    parser.add_argument("--device", default=7, type=int)
    parser.add_argument("--kld_coeff", default=0.5, type=float)
    parser.add_argument("--future_loss_coeff", default=1, type=float)
    parser.add_argument("--dest_loss_coeff", default=2, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--lr_decay_step_size", default=10, type=int)
    parser.add_argument("--lr_decay_gamma", default=0.5, type=float)
    parser.add_argument("--mu", default=0, type=float)
    parser.add_argument("--n_values", default=20, type=int)
    parser.add_argument("--nonlocal_pools", default=3, type=int)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--past_length", default=8, type=int)
    parser.add_argument("--sigma", default=1.3, type=float)
    parser.add_argument("--zdim", default=16, type=int)
    parser.add_argument("--print_log", default=6, type=int)
    parser.add_argument("--sub_goal_indexes", default=[2, 5, 8, 11], type=list)


    parser.add_argument('--e_prior_sig', type=float, default=2, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=2, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_steps_pcd', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')
    parser.add_argument('--e_lr', default=0.00003, type=float)
    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--e_max_norm', type=float, default=25, help='max norm allowed')
    parser.add_argument('--e_decay', default=1e-4, help='weight decay for ebm')
    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--e_beta1', default=0.9, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)
    parser.add_argument('--memory_size', default=200000, type=int)


    parser.add_argument('--dataset_name', type=str, default='zara2')
    parser.add_argument('--save_folder', type=str, default='1104_new_DE/')
    parser.add_argument('--save_config', type=str, default='newDE2/')
    parser.add_argument('--dataset_folder', type=str, default='dataset')
    parser.add_argument('--newDE', default=True, type=bool)
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--batch_size',type=int,default=70)


    parser.add_argument('--ny', type=int, default=1)
    # parser.add_argument('--model_path', type=str, default='saved_models/lbebm_eth.pt')
    parser.add_argument('--model_path', type=str, default=None)

    return parser.parse_args()


def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def main():

    # exp_id = get_exp_id(__file__)
    # output_dir = get_output_dir(exp_id)
    # copy_source(__file__, output_dir)

    args = parse_args()
    # custom
    # if(args.dataset_name=='eth'):
    #     args.seed=3
    #     args.kld_coeff=0.5
    #     args.lr_decay_step_size=10
    # elif(args.dataset_name=='hotel'):
    #     args.seed=2
    #     args.kld_coeff=0.8
    #     args.lr_decay_step_size=30
    # elif(args.dataset_name=='univ'): 
    #     args.seed=1
    #     args.kld_coeff=0.5
    #     args.lr_decay_step_size=30
    # elif(args.dataset_name=='zara1'): 
    #     args.seed=1
    #     args.kld_coeff=0.5
    #     args.lr_decay_step_size=30
    # elif(args.dataset_name=='zara2'): 
    #     args.seed=1
    #     args.kld_coeff=0.5
    #     args.lr_decay_step_size=30
    
    output_dir='/home/yaoliu/scratch/experiment/lbebm/'+args.save_folder + args.save_config + args.dataset_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    copy_source(__file__, output_dir)
    # set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)
    args.way_points = list(set(list(range(args.future_length))) - set(args.sub_goal_indexes))

    logger = setup_logging('job{}'.format(0), output_dir, console=True)
    logger.info(args)

    if args.val_size==0:
        train_dataset, _ = utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True, verbose=True)
        val_dataset, _ = utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False, verbose=True)
    else:
        train_dataset, val_dataset = utils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs, args.preds, delim=args.delim, train=True, verbose=args.verbose)

    test_dataset, _ =  utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True, verbose=True)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size*10, shuffle=False, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*10, shuffle=False, num_workers=0)

    def initial_pos(traj_batches):
        batches = []
        for b in traj_batches:
            starting_pos = b[:,7,:].copy()/1000
            batches.append(starting_pos)
        return batches

    def sample_p_0(n, nz=16):
        return args.e_init_sig * torch.randn(*[n, nz]).double().cuda()

    def calculate_loss(dest, dest_recon, mean, log_var, criterion, future, interpolated_future, sub_goal_indexes):
        dest_loss = criterion(dest, dest_recon)
        future_loss = criterion(future, interpolated_future)
        subgoal_reg = criterion(dest_recon, interpolated_future.view(dest.size(0), future.size(1)//2, 2)[:, sub_goal_indexes, :].view(dest.size(0), -1))
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return dest_loss, future_loss, kl, subgoal_reg

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()

            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + self.pe[: x.size(0), :]
            return self.dropout(x)

    class ConcatSquashLinear(nn.Module):
        def __init__(self, dim_in, dim_out, dim_ctx):
            super(ConcatSquashLinear, self).__init__()
            self._layer = Linear(dim_in, dim_out)
            self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
            self._hyper_gate = Linear(dim_ctx, dim_out)

        def forward(self, ctx, x):
            gate = torch.sigmoid(self._hyper_gate(ctx))
            bias = self._hyper_bias(ctx)
            # if x.dim() == 3:
            #     gate = gate.unsqueeze(1)
            #     bias = bias.unsqueeze(1)
            ret = self._layer(x) * gate + bias
            return ret        

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
            super(MLP, self).__init__()
            dims = []
            dims.append(input_dim)
            dims.extend(hidden_size)
            dims.append(output_dim)
            self.layers = nn.ModuleList()
            for i in range(len(dims)-1):
                self.layers.append(nn.Linear(dims[i], dims[i+1]))

            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()

            self.sigmoid = nn.Sigmoid() if discrim else None
            self.dropout = dropout

        def forward(self, x):
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i != len(self.layers)-1:
                    x = self.activation(x)
                    if self.dropout != -1:
                        x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
                elif self.sigmoid:
                    x = self.sigmoid(x)
            return x

    class ReplayMemory(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, input_memory):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = input_memory
            self.position = (self.position + 1) % self.capacity

        def sample(self, n=100):
            samples = random.sample(self.memory, n)
            return torch.cat(samples)

        def __len__(self):
            return len(self.memory)

    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    class LBEBM(nn.Module):
        def __init__(self, 
                    enc_past_size, 
                    enc_dest_size, 
                    enc_latent_size, 
                    dec_size, 
                    predictor_size, 
                    fdim, 
                    zdim, 
                    sigma, 
                    past_length, 
                    future_length):
            super(LBEBM, self).__init__()

            # var_sched
            self.num_steps = 10
            self.beta_1 = 1e-4
            self.beta_T = 5e-2
            self.mode = 'linear'
            self.cosine_s=8e-3

            if self.mode == 'linear':
                betas = torch.linspace(self.beta_1, self.beta_T, steps=self.num_steps)
            elif self.mode == 'cosine':
                timesteps = (
                torch.arange(self.num_steps + 1) / self.num_steps + self.cosine_s
                )
                alphas = timesteps / (1 + self.cosine_s) * math.pi / 2
                alphas = torch.cos(alphas).pow(2)
                alphas = alphas / alphas[0]
                betas = 1 - alphas[1:] / alphas[:-1]
                betas = betas.clamp(max=0.999)

            betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

            alphas = 1 - betas
            log_alphas = torch.log(alphas)
            for i in range(1, log_alphas.size(0)):  # 1 to T
                log_alphas[i] += log_alphas[i - 1]
            alpha_bars = log_alphas.exp()

            sigmas_flex = torch.sqrt(betas)
            sigmas_inflex = torch.zeros_like(sigmas_flex)
            for i in range(1, sigmas_flex.size(0)):
                sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
            sigmas_inflex = torch.sqrt(sigmas_inflex)

            self.register_buffer('betas', betas)
            self.register_buffer('alphas', alphas)
            self.register_buffer('alpha_bars', alpha_bars)
            self.register_buffer('sigmas_flex', sigmas_flex)
            self.register_buffer('sigmas_inflex', sigmas_inflex)

            # backbone
            # point_dim=2
            context_dim=256
            tf_layer=3
            residual=False
            d_model=4
            self.residual = residual
            self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=20)
            self.concat1 = ConcatSquashLinear(zdim,2*zdim,fdim+3)
            self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=zdim)
            self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
            self.concat3 = ConcatSquashLinear(2*zdim,4*zdim,fdim+3)
            self.concat4 = ConcatSquashLinear(4*zdim,2*zdim,fdim+3)
            self.linear = ConcatSquashLinear(2*zdim, zdim, fdim+3)

            self.zdim = zdim
            self.sigma = sigma
            self.nonlocal_pools = args.nonlocal_pools
            non_local_dim = args.non_local_dim
            non_local_phi_size = args.non_local_phi_size
            non_local_g_size = args.non_local_g_size
            non_local_theta_size = args.non_local_theta_size

            self.encoder_past = MLP(input_dim=past_length*2, output_dim=fdim, hidden_size=enc_past_size)
            self.encoder_dest = MLP(input_dim=len(args.sub_goal_indexes)*2, output_dim=fdim, hidden_size=enc_dest_size)
            self.encoder_latent = MLP(input_dim=2*fdim, output_dim=2*zdim, hidden_size=enc_latent_size)
            self.decoder = MLP(input_dim=fdim+zdim, output_dim=len(args.sub_goal_indexes)*2, hidden_size=dec_size)
            self.predictor = MLP(input_dim=2*fdim, output_dim=2*(future_length), hidden_size=predictor_size)

            self.non_local_theta = MLP(input_dim = fdim, output_dim = non_local_dim, hidden_size=non_local_theta_size)
            self.non_local_phi = MLP(input_dim = fdim, output_dim = non_local_dim, hidden_size=non_local_phi_size)
            self.non_local_g = MLP(input_dim = fdim, output_dim = fdim, hidden_size=non_local_g_size)

            self.EBM = nn.Sequential(
                nn.Linear(zdim + fdim, 200),
                nn.GELU(),
                nn.Linear(200, 200),
                nn.GELU(),
                nn.Linear(200, args.ny),
                )
                        
            self.replay_memory = ReplayMemory(args.memory_size)

        def forward(self, x, dest=None, mask=None, iteration=1, y=None):
            # x torch.Size([70, 16])
            ftraj = self.encoder_past(x) # torch.Size([70, 16])

            if mask:
                for _ in range(self.nonlocal_pools):
                    ftraj = self.non_local_social_pooling(ftraj, mask)

            if self.training:
                dest_features = self.encoder_dest(dest) # torch.Size([70, 16])
                features = torch.cat((ftraj, dest_features), dim=1) # torch.Size([70, 32])
                latent =  self.encoder_latent(features) # torch.Size([70, 32])
                mu = latent[:, 0:self.zdim]
                logvar = latent[:, self.zdim:]

                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_().cuda()
                z_g_k = eps.mul(var).add_(mu)
                z_g_k = z_g_k.double().cuda() # torch.Size([70, 16])

            if self.training:
                # pcd = True if len(self.replay_memory) == args.memory_size else False
                # if pcd:
                #     z_e_0 = self.replay_memory.sample(n=ftraj.size(0)).clone().detach().cuda()
                # else:
                #     z_e_0 = sample_p_0(n=ftraj.size(0), nz=self.zdim)
                # z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=pcd, verbose=(iteration % 1000==0))
                # for _z_e_k in z_e_k.clone().detach().cpu().split(1):
                #     self.replay_memory.push(_z_e_k)
                z_e_k = z_g_k
            else:
                # z_e_0 = sample_p_0(n=ftraj.size(0), nz=self.zdim) # torch.Size([70, 16])
                z_e_k = self.diffusion_sample_ddpm(ftraj)
                # z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=False, verbose=(iteration % 1000==0), y=y)                        
            z_e_k = z_e_k.double().cuda()

            if self.training:
                decoder_input = torch.cat((ftraj, z_g_k), dim=1)
            else:
                decoder_input = torch.cat((ftraj, z_e_k), dim=1)
            generated_dest = self.decoder(decoder_input)

            if self.training:
                generated_dest_features = self.encoder_dest(generated_dest)
                prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
                pred_future = self.predictor(prediction_features)

                # en_pos = self.ebm(z_g_k, ftraj).mean() # torch.Size([70]) mean
                # en_neg = self.ebm(z_e_k.detach().clone(), ftraj).mean()
                # cd = en_pos - en_neg
                cd = self.diffusion_loss(z_g_k, ftraj)
                return generated_dest, mu, logvar, pred_future, cd#, en_pos, en_neg, pcd

            return generated_dest

        def diffusion_loss(self, x_0, context, t=None):
            # x_0 70,16

            batch_size, point_dim = x_0.size()
            if t == None:
                t = self.uniform_sample_t(batch_size)

            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t].cuda()

            c0 = torch.sqrt(alpha_bar).view(-1, 1).cuda()       # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1).cuda()   # (B, 1, 1)

            e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)

            e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
            loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
            return loss

        def diffusion_sample(self, context):

            self.alphas_cumprod = self.alpha_bars

            batch_size = context.size(0)

            ddim_timesteps=10
            ddim_eta=0.0
            clip_denoised=False

            c = self.num_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.num_steps, c)))
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            ddim_timestep_seq = ddim_timestep_seq + 1
            # previous sequence
            ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

            # sample_img = torch.randn([batch_size, self.zdim]).to(context.device)
            sample_img = sample_p_0(n=context.size(0), nz=self.zdim)

            for i in reversed(range(0, ddim_timesteps)) :
                t = torch.full((batch_size,), ddim_timestep_seq[i], device=context.device, dtype=torch.long)
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=context.device, dtype=torch.long)
                
                # 1. get current and previous alpha_cumprod
                
                alpha_cumprod_t = extract(self.alphas_cumprod, t, sample_img.shape)
                alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, sample_img.shape)
        
                # 2. predict noise using model
                beta = self.betas[[t[0].item()]*batch_size]
                pred_noise = self.net(sample_img, beta=beta, context=context)
                
                # 3. get the predicted x_0
                pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                if clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                
                # 4. compute variance: "sigma_t(η)" -> see formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
                
                # 5. compute "direction pointing to x_t" of formula (12)
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
                
                # 6. compute x_{t-1} of formula (12)
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

                sample_img = x_prev.detach()

            return sample_img

        def diffusion_sample_ddpm(self, context):
            batch_size = context.size(0)

            x_T = sample_p_0(n=context.size(0), nz=self.zdim)

            traj = {self.num_steps: x_T} # {100:xt}
            for t in range(self.num_steps, 0, -1):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                sigma = self.get_sigmas(t, 0.0)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.betas[[t]*batch_size]
                e_theta = self.net(x_t, beta=beta, context=context)
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                del traj[t]

            return traj[0]

        def net(self, x, beta, context):
            batch_size = x.size(0)
            beta = beta.view(batch_size, 1)          # (B, 1, 1)
            context = context.view(batch_size, -1)   # (B, 1, F)

            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
            ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
            x = self.concat1(ctx_emb,x)
            final_emb = x.reshape(x.size()[0],8,-1).permute(1,0,2)

            final_emb = self.pos_emb(final_emb)

            trans = self.transformer_encoder(final_emb).permute(1,0,2).reshape(x.size()[0],-1)
            trans = self.concat3(ctx_emb, trans)
            trans = self.concat4(ctx_emb, trans)
            return self.linear(ctx_emb, trans)

        def uniform_sample_t(self, batch_size):
            ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
            return ts.tolist()

        def get_sigmas(self, t, flexibility):
            assert 0 <= flexibility and flexibility <= 1
            sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
            return sigmas

        def ebm(self, z, condition, cls_output=False):
            condition_encoding = condition.detach().clone()
            z_c = torch.cat((z, condition_encoding), dim=1)
            conditional_neg_energy = self.EBM(z_c)
            assert conditional_neg_energy.shape == (z.size(0), args.ny)
            if cls_output:
                return - conditional_neg_energy
            else:
                return - conditional_neg_energy.logsumexp(dim=1)
        
        def sample_langevin_prior_z(self, z, condition, pcd=False, verbose=False, y=None):
            z = z.clone().detach()
            z.requires_grad = True
            _e_l_steps = args.e_l_steps_pcd if pcd else args.e_l_steps
            _e_l_step_size = args.e_l_step_size
            for i in range(_e_l_steps):
                if y is None:
                    en = self.ebm(z, condition)
                else:
                    en = self.ebm(z, condition, cls_output=True)[range(z.size(0)), y]
                z_grad = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * _e_l_step_size * _e_l_step_size * (z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
                if args.e_l_with_noise:
                    z.data += _e_l_step_size * torch.randn_like(z).data

                if (i % 5 == 0 or i == _e_l_steps - 1) and verbose:
                    if y is None:
                        print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, _e_l_steps, en.sum().item()))
                    else:
                        logger.info('Conditional Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i + 1, _e_l_steps, en.sum().item()))

                z_grad_norm = z_grad.view(z_grad.size(0), -1).norm(dim=1).mean()

            return z.detach(), z_grad_norm

        def predict(self, past, generated_dest):
            ftraj = self.encoder_past(past)
            generated_dest_features = self.encoder_dest(generated_dest)
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)
            interpolated_future = self.predictor(prediction_features)

            return interpolated_future

        def non_local_social_pooling(self, feat, mask):
            theta_x = self.non_local_theta(feat)
            phi_x = self.non_local_phi(feat).transpose(1,0)
            f = torch.matmul(theta_x, phi_x)
            f_weights = F.softmax(f, dim = -1)
            f_weights = f_weights * mask
            f_weights = F.normalize(f_weights, p=1, dim=1)
            pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

            return pooled_f + feat
    
    def train(model, optimizer, epoch, sub_goal_indexes):
        model.train()
        train_loss, total_dest_loss, total_future_loss = 0, 0, 0
        criterion = nn.MSELoss()

        for i, trajx in enumerate(tr_dl):
            x = trajx['src'][:, :, :2]
            y = trajx['trg'][:, :, :2]
            x = x - trajx['src'][:, -1:, :2]
            y = y - trajx['src'][:, -1:, :2]

            x *= args.data_scale
            y *= args.data_scale
            x = x.double().cuda()
            y = y.double().cuda()

            x = x.view(-1, x.shape[1]*x.shape[2])
            dest = y[:, sub_goal_indexes, :].detach().clone().view(y.size(0), -1)
            future = y.view(y.size(0),-1)

            dest_recon, mu, var, interpolated_future, cd= model.forward(x, dest=dest, mask=None, iteration=i)

            optimizer.zero_grad()
            dest_loss, future_loss, kld, subgoal_reg = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future, sub_goal_indexes)
            loss = args.dest_loss_coeff * dest_loss + args.future_loss_coeff * future_loss + args.kld_coeff * kld  + cd*10 + subgoal_reg
            # loss = args.dest_loss_coeff * dest_loss + args.future_loss_coeff * future_loss + args.kld_coeff * kld  + cd + subgoal_reg
            # loss = dest_loss + future_loss + kld  + cd + subgoal_reg
            loss.backward()

            train_loss += loss.item()
            total_dest_loss += dest_loss.item()
            total_future_loss += future_loss.item()
            optimizer.step()

            if (i+1) % args.print_log == 0:
                logger.info('{:5d}/{:5d} '.format(i, epoch) +
                            'dest_loss={:8.6f} '.format(dest_loss.item()) +
                            'future_loss={:8.6f} '.format(future_loss.item()) +
                            'kld={:8.6f} '.format(kld.item()) +
                            'cd={:8.6f} '.format(cd.item()) +
                            'subgoal_reg={}'.format(subgoal_reg.detach().cpu().numpy())
                )

        return train_loss, total_dest_loss, total_future_loss

    def test(model, dataloader, dataset, sub_goal_indexes, best_of_n=20):
        model.eval()

        total_dest_err = 0.
        total_overall_err = 0.

        for i, trajx in enumerate(dataloader):
            x = trajx['src'][:, :, :2]
            y = trajx['trg'][:, :, :2]
            x = x - trajx['src'][:, -1:, :2]
            y = y - trajx['src'][:, -1:, :2]

            x *= args.data_scale
            y *= args.data_scale
            x = x.double().cuda()
            y = y.double().cuda()

            y = y.cpu().numpy()

            x = x.view(-1, x.shape[1]*x.shape[2])

            plan = y[:, sub_goal_indexes, :].reshape(y.shape[0],-1)

            if(args.newDE):
                plan_target = plan
                gt = y

                all_plans = np.zeros((best_of_n, plan_target.shape[0], plan_target.shape[1]))
                all_plan_errs = np.zeros((best_of_n, plan_target.shape[0], plan_target.shape[1] // 2))
                for i in range(best_of_n):
                    plan_pred = model.forward(x, mask=None)  # 注意：这里假设您的模型可以接受numpy数组作为输入
                    plan_pred=plan_pred.detach().cpu().numpy()
                    all_plans[i] = plan_pred
                    l2_distances_list = []
                    for idx in range(0, plan_target.shape[1], 2):
                        l2_distance = np.linalg.norm(all_plans[i][:, idx:idx+2] - plan_target[:, idx:idx+2], axis=1)
                        l2_distances_list.append(l2_distance)
                    l2_distances = np.stack(l2_distances_list, axis=1)
                    all_plan_errs[i] = l2_distances

                selected_plans_list = []
                selected_error_list = []
                for i in range(all_plan_errs.shape[2]):
                    best_index = np.argmin(all_plan_errs[:, :, i], axis=0)
                    selected_plan_this = all_plans[best_index, np.arange(best_index.shape[0]), 2*i:2*(i+1)]
                    selected_plans_list.append(selected_plan_this)

                    selected_error_this = all_plan_errs[best_index, np.arange(best_index.shape[0]), i]
                    selected_error_list.append(selected_error_this)

                selected_plan = np.stack(selected_plans_list, axis=1)
                selected_plan = selected_plan.reshape(selected_plan.shape[0], -1)
                selected_error = np.stack(selected_error_list, axis=1)
                best_dest_err = np.sum(selected_error[:,-1])

                interpolated_future = model.predict(x, torch.DoubleTensor(selected_plan).cuda())  # 注意：这里假设您的模型可以接受numpy数组作为输入
                interpolated_future=interpolated_future.detach().cpu().numpy()
                interpolated_future = interpolated_future.reshape(-1, args.future_length, 2)
                l2_distance_avg = np.linalg.norm(interpolated_future - gt, axis=2)

                selected_distances_plan_index = np.array(args.sub_goal_indexes)
                # selected_distances_plan = l2_distance_avg[:, selected_distances_plan_index]
                # min_values = np.minimum(selected_distances_plan, selected_error)
                # l2_distance_avg[:, selected_distances_plan_index] = min_values

                l2_distance_avg[:, selected_distances_plan_index]=selected_error

                overall_err = np.sum(np.mean(l2_distance_avg, axis=1))

                overall_err /= args.data_scale
                best_dest_err /= args.data_scale

                total_overall_err += overall_err
                total_dest_err += best_dest_err
            else:    
                all_plan_errs = []
                all_plans = []
                for _ in range(best_of_n):
                    # dest_recon = model.forward(x, initial_pos, device=device)
                    # modes = torch.tensor(k % args.ny, device=device).long().repeat(batch_size)
                    plan_recon = model.forward(x, mask=None)
                    plan_recon = plan_recon.detach().cpu().numpy()
                    all_plans.append(plan_recon)
                    plan_err = np.linalg.norm(plan_recon - plan, axis=-1)
                    all_plan_errs.append(plan_err)

                all_plan_errs = np.array(all_plan_errs) 
                all_plans = np.array(all_plans) 
                indices = np.argmin(all_plan_errs, axis=0)
                best_plan = all_plans[indices, np.arange(x.shape[0]),  :]

                # FDE
                best_dest_err = np.linalg.norm(best_plan[:, -2:] - plan[:, -2:], axis=1).sum()

                best_plan = torch.DoubleTensor(best_plan).cuda()
                interpolated_future = model.predict(x, best_plan)
                interpolated_future = interpolated_future.detach().cpu().numpy()

                # ADE        
                predicted_future = np.reshape(interpolated_future, (-1, args.future_length, 2))
                overall_err = np.linalg.norm(y - predicted_future, axis=-1).mean(axis=-1).sum()

                overall_err /= args.data_scale
                best_dest_err /= args.data_scale

                total_overall_err += overall_err
                total_dest_err += best_dest_err

        total_overall_err /= len(dataset)
        total_dest_err /= len(dataset)

        return total_overall_err, total_dest_err

    def run_training(args):
        model = LBEBM(
            args.enc_past_size,
            args.enc_dest_size,
            args.enc_latent_size,
            args.dec_size,
            args.predictor_hidden_size,
            args.fdim,
            args.zdim,
            args.sigma,
            args.past_length,
            args.future_length)
        
        model = model.double().cuda()
        optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_gamma)

        best_val_ade = 50
        best_val_fde = 50
        best_test_ade = 50
        best_test_fde = 50

        patience_epoch = 0

        for epoch in range(args.num_epochs):
            train_loss, dest_loss, overall_loss = train(model, optimizer, epoch, args.sub_goal_indexes)
            overall_err, dest_err = test(model, val_dl, val_dataset, args.sub_goal_indexes, args.n_values)
            test_overall_err, test_dest_err = test(model, test_dl, test_dataset, args.sub_goal_indexes, args.n_values)

            patience_epoch += 1
            if best_val_ade > overall_err:
                patience_epoch = 0
                best_val_ade = overall_err
                best_val_fde = dest_err

            if best_test_ade > test_overall_err:
                best_test_ade = test_overall_err
                best_test_fde = test_dest_err

            logger.info("Train Loss {}".format(train_loss))
            logger.info("Overall Loss {}".format(overall_loss))
            logger.info("Dest Loss {}".format(dest_loss))
            logger.info("Val ADE {}".format(overall_err))
            logger.info("Val FDE {}".format(dest_err))
            logger.info("Val Best ADE {}".format(best_val_ade))
            logger.info("Val Best FDE {}".format(best_val_fde))
            logger.info("Test ADE {}".format(test_overall_err))
            logger.info("Test FDE {}".format(test_dest_err))
            logger.info("Test Best ADE {}".format(best_test_ade))
            logger.info("Test Best FDE {}".format(best_test_fde))
            logger.info("----->learning rate {}".format(optimizer.param_groups[0]['lr'])) 

            scheduler.step()

    def run_eval(args):
        model = LBEBM(
            args.enc_past_size,
            args.enc_dest_size,
            args.enc_latent_size,
            args.dec_size,
            args.predictor_hidden_size,
            args.fdim,
            args.zdim,
            args.sigma,
            args.past_length,
            args.future_length)
        
        model = model.double().cuda()
        
        ckpt = torch.load(args.model_path, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt['model_state_dict'])
        overall_err, dest_err = test(model, test_dl, test_dataset, args.sub_goal_indexes, args.n_values)
        logger.info("Test ADE {}".format(overall_err))
        logger.info("Test FDE {}".format(dest_err))

    if args.model_path:
        run_eval(args)
    else:
        run_training(args)

if __name__ == '__main__':
    main()