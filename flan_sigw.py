import torch
import numpy as np
import scipy
#we disable auto-differentiation to reduce memory usage
torch.set_grad_enabled(False)


def build_log_normal(A_star, k_star, sigma_star): 
    def Delta_zeta(k): 
        if torch.is_tensor(k):
            denom = np.sqrt(2. * np.pi * sigma_star**2)
            return A_star * torch.exp(-torch.log(k/k_star)**2 / (2. * sigma_star**2)) / denom
        else:
            return A_star * np.exp(-np.log(k/k_star)**2 / (2. *sigma_star**2)) / np.sqrt(2. * np.pi * sigma_star**2)
    return Delta_zeta

def T_rad(q, s):
    return 12.*(q**2 -1)**2*(s**2 - 1)**2*(q**2 + s**2 - 6)**4/(s**2 - q**2)**8*((np.log((3 - q**2)/np.fabs(3 - s**2)) + 2*(s**2 - q**2)/(q**2 + s**2 - 6))**2 + np.pi**2*np.heaviside(s - np.sqrt(3), 0.5)) 

def Omega_gw_semianalitic(k, Delta_zeta):
    q = np.arange(0., 0.999, 0.01) 
    s  = 10**np.arange(0, 3, 0.005)  
    y = T_rad(q[:, None, None], s[None, :, None])*Delta_zeta(0.5*k[None, None, :]*(q[:, None, None] + s[None, :, None]))*Delta_zeta(0.5*k[None, None, :]*(s[None, :, None]-q[:, None, None]))
    return scipy.integrate.trapezoid(scipy.integrate.trapezoid(y, s, axis = 1), q, axis = 0)
    
def generate_gaussian_field_x(N, L, Delta_X, device=torch.device('cpu')):
    dx3 = (L/N)**3
    P_X = lambda k: Delta_X(k)*2.*np.pi**2/k**3
    q_whole = 2*torch.pi*torch.fft.fftfreq(N, d=L/N, device=device)
    q_half  = 2*torch.pi*torch.fft.rfftfreq(N, d=L/N, device=device)
    qx, qy, qz = torch.meshgrid(q_whole, q_whole, q_half, indexing='ij')
    q_grid = torch.sqrt(qx**2 + qy**2 + qz**2)

    X_wn = (N / L)**1.5 * torch.randn((N, N, N), device=device, dtype=torch.float32)
    X_k = torch.fft.rfftn(X_wn, s=(N, N, N), dim=(0, 1, 2))*dx3
    X_k[q_grid > 0] = X_k[q_grid > 0] * torch.sqrt(P_X(q_grid[q_grid > 0]))
    X_k[q_grid == 0] = 0
   
    return torch.fft.irfftn(X_k, s=(N, N, N), dim=(0, 1, 2))/dx3

def generate_gaussian_field_k(N, L, Delta_X, device=torch.device('cpu')):
    dx3 = (L/N)**3
    P_X = lambda k: Delta_X(k)*2.*np.pi**2/k**3
    q_whole = 2*torch.pi*torch.fft.fftfreq(N, d=L/N, device=device)
    q_half  = 2*torch.pi*torch.fft.rfftfreq(N, d=L/N, device=device)
    qx, qy, qz = torch.meshgrid(q_whole, q_whole, q_half, indexing='ij')
    q_grid = torch.sqrt(qx**2 + qy**2 + qz**2)

    X_wn = (N / L)**1.5 * torch.randn((N, N, N), device=device, dtype=torch.float32)
    if device.type == 'xpu':
        torch.xpu.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
     
    X_k = torch.fft.rfftn(X_wn, s=(N, N, N), dim=(0, 1, 2))*dx3
    X_k[q_grid > 0] = X_k[q_grid > 0] * torch.sqrt(P_X(q_grid[q_grid > 0]))
    X_k[q_grid == 0] = 0
    
    return X_k

def generate_nongaussian_field_k(N, L, Delta_X_gauss, F_NL, G_NL, device = torch.device('cpu')):
    X_gauss = generate_gaussian_field_x(N, L, Delta_X_gauss, device = device)
    X = X_gauss + F_NL*X_gauss**2 + G_NL*X_gauss**3
    dx3 = (L/N)**3   
    X_k = torch.fft.rfftn(X, s=(N, N, N), dim=(0, 1, 2))*dx3
    return X_k


def compute_laplacian_x(N, L, X, device = torch.device('cpu')): 
    q_whole = 2*torch.pi*torch.fft.fftfreq(N, d=L/N, device=device)
    q_half  = 2*torch.pi*torch.fft.rfftfreq(N, d=L/N, device=device)
    qx, qy, qz = torch.meshgrid(q_whole, q_whole, q_half, indexing='ij')
    q_grid = torch.sqrt(qx**2 + qy**2 + qz**2)
    #since we transform and immediately transform back we can avoid to normalize by dx^3
    X_k = torch.fft.rfftn(X, s=(N, N, N), dim=(0, 1, 2))
    return torch.fft.irfftn( - q_grid**2*X_k, s=(N, N, N), dim=(0, 1, 2)).real 

def generate_nongaussian_nonlocal_field_k(N, L, Delta_X_gauss, k_star, alpha_NL, beta_NL, device = torch.device('cpu')):
    X_gauss = generate_gaussian_field_x(N, L, Delta_X_gauss, device = device)
    X = X_gauss + (alpha_NL/k_star**2)*compute_laplacian_x(N, L, X_gauss**2, device = device)
    X += (beta_NL/k_star**2)*X_gauss*compute_laplacian_x(N, L, X_gauss, device = device)
    dx3 = (L/N)**3   
    X_k = torch.fft.rfftn(X, s=(N, N, N), dim=(0, 1, 2))*dx3
    return X_k



def generate_nongaussian_field_log_k(N, L, Delta_X_gauss, mu, device = torch.device('cpu')):
    X_gauss = generate_gaussian_field_x(N, L, Delta_X_gauss, device = device)
    X = -mu*torch.log(torch.abs(1. - X_gauss/mu))
    dx3 = (L/N)**3   
    X_k = torch.fft.rfftn(X, s=(N, N, N), dim=(0, 1, 2))*dx3
    return X_k

#estimates the errors with Jackknife method
def compute_power_spectrum_k(k, X_k, L, Delta=1, n_blocks_1d=4):
    device = X_k.device
    N = len(X_k[:, 0, 0])
    
    k_min = 2. * torch.pi / L
    kx = torch.fft.fftfreq(N, d=L/N, device=device) * 2 * torch.pi
    kz_r = torch.fft.rfftfreq(N, d=L/N, device=device) * 2 * torch.pi
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, kx, kz_r, indexing='ij')
    k_grid = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    delta_k = Delta * k_min
    mask = (k_grid < k + delta_k/2.) & (k_grid > k - delta_k/2.)
    
    P_bins_full = torch.abs(X_k[mask]).flatten()**2 / L**3
    P_central = torch.mean(P_bins_full)
    
    X_real = torch.fft.irfftn(X_k, s=(N, N, N))
    
    block_size = N // n_blocks_1d
    M = n_blocks_1d**3
    P_replicas = torch.zeros(M, device=device)
    volume_fraction = (M - 1) / M
    
    idx = 0
    for i in range(n_blocks_1d):
        for j in range(n_blocks_1d):
            for l in range(n_blocks_1d):
                spatial_mask = torch.ones_like(X_real)

                x_start, x_end = i*block_size, (i+1)*block_size
                y_start, y_end = j*block_size, (j+1)*block_size
                z_start, z_end = l*block_size, (l+1)*block_size
                spatial_mask[x_start:x_end, y_start:y_end, z_start:z_end] = 0.0
                
                X_real_masked = X_real * spatial_mask
                X_k_replica = torch.fft.rfftn(X_real_masked)
                
                P_bins = torch.abs(X_k_replica[mask])**2 / (L**3 * volume_fraction)
                P_replicas[idx] = torch.mean(P_bins)
                idx += 1
                
    mean_P = torch.mean(P_replicas)
    std_P = torch.sqrt((M - 1) * torch.mean((P_replicas - mean_P)**2))
    
    return P_central.cpu().numpy(), std_P.cpu().numpy()

def I_c(u, v, eps=0.): 
    res = (u**2 + v**2 - 3)*(-4.*u*v + (u**2 + v**2 - 3)*torch.log(torch.abs(((3 - (u + v)**2)**2 + eps**2)**0.5/(3 - (u - v)**2))))
    return (27.0 / 32.0) * res / (u**3 * v**3)

def I_s(u, v, eps=0.): 
    res = np.pi*(u**2 + v**2 - 3)**2 * torch.heaviside(u + v - 3**0.5, torch.tensor(0., device=u.device))
    return (27.0 / 32.0) * res / (u**3 * v**3)

#given the number of lattice spacings per side, and the side length, computed the polarization tensors
#threshold is a number in (0.5, 1) which can be chosen in the definition of the basis. 
#in practice, if \hat k has a y component smaller than threshold, then u_hat = e_y\cross k_hat; 
#if larger, u_hat = e_x \cross k_hat. In fact, for k_hat = e_y, u_hat = 0, nonsensical.
#I choose a variable threshold to check that the final result doesn't depend on its value.
#notice that because of hairy ball theorem, there is not a continuous way to map hat k to hat u!

def compute_polarization_basis(N, L, device, threshold=0.5):
    k_whole = 2*torch.pi*torch.fft.fftfreq(N, d=L/N, device=device)
    k_half = 2*torch.pi*torch.fft.rfftfreq(N, d=L/N, device=device)
    kx, ky, kz = torch.meshgrid(k_whole, k_whole, k_half, indexing='ij')
    k_vec = torch.stack([kx, ky, kz], dim=0)
    k_grid = torch.sqrt(torch.sum(k_vec**2, dim=0))
    k_hat = torch.zeros_like(k_vec, device=device)

    k_mask = k_grid > 0
    k_hat[:, k_mask] = k_vec[:, k_mask] / k_grid[k_mask]
    e_x = torch.tensor([1., 0., 0.], device=device).view(3, 1, 1, 1)
    e_y = torch.tensor([0., 1., 0.], device=device).view(3, 1, 1, 1)
    
    u_hat = torch.zeros_like(k_hat, device=device)
    y_mask = torch.abs(k_hat[1]) < threshold         
    u_hat[:, y_mask] = torch.linalg.cross(e_y.expand_as(k_hat)[:, y_mask], k_hat[:, y_mask], dim=0)
    u_hat[:, ~y_mask] = torch.linalg.cross(e_x.expand_as(k_hat)[:, ~y_mask], k_hat[:, ~y_mask], dim=0)
    
    u_mag = torch.norm(u_hat, dim=0, keepdim=True)
    u_mask = u_mag > 0
    u_hat = torch.where(u_mask, u_hat / u_mag, torch.zeros_like(u_hat))
    v_hat = torch.linalg.cross(k_hat, u_hat, dim=0)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
        
    e_plus = inv_sqrt2 * (u_hat[:, None, ...] * u_hat[None, :, ...] - 
                          v_hat[:, None, ...] * v_hat[None, :, ...])
        
    e_cross = inv_sqrt2 * (u_hat[:, None, ...] * v_hat[None, :, ...] + 
                           v_hat[:, None, ...] * u_hat[None, :, ...])
    return k_grid, e_plus, e_cross

def create_smooth_density_grid(u_min=1e-2, u_max=15.0, N_modes=150, u_pivot=1, sigma=1, amplitude=10.0, device=torch.device('cpu')):
    u_dense = np.linspace(u_min, u_max, 5000)
    rho = 1.0 + amplitude * np.exp(-0.5 * ((u_dense - u_pivot) / sigma)**2)
    n_cum = scipy.integrate.cumulative_trapezoid(rho, u_dense, initial=0)
    
    n_cum_normalized = n_cum / n_cum[-1] * (N_modes - 1)
    n_integer_indices = np.arange(N_modes)
    u_grid = np.interp(n_integer_indices, n_cum_normalized, u_dense)
    u_tensor = torch.tensor(u_grid, dtype=torch.float32, device=device)
    
    w = torch.zeros_like(u_tensor)
    w[1:-1] = (u_tensor[2:] - u_tensor[:-2]) / 2.0
    w[0] = (u_tensor[1] - u_tensor[0]) / 2.0
    w[-1] = (u_tensor[-1] - u_tensor[-2]) / 2.0
    
    return u_tensor, w

def build_indicator_evaluator(u_centers, w_widths, phi_amplitudes):
    device = u_centers.device 
    N = len(u_centers)
    
    edges = torch.zeros(N + 1, dtype=u_centers.dtype, device=device)
    edges[1:-1] = (u_centers[1:] + u_centers[:-1]) / 2.0
    edges[0] = u_centers[0] - w_widths[0] / 2.0 
    edges[-1] = u_centers[-1] + w_widths[-1] / 2.0

    def evaluate(x):
        mu_indices = torch.bucketize(x, edges) - 1
        mu_indices = torch.clamp(mu_indices, 0, N - 1)
        return phi_amplitudes[mu_indices]
        
    return evaluate

#takes as input the grid and associated weights

def decompose_kernels(u, w): 
    U, V = torch.meshgrid(u, u, indexing='ij')
    mask = (torch.abs(U - V) <= 1) & (U + V >= 1) 
    W_sqrt = torch.sqrt(w)
    W_mat = torch.outer(W_sqrt, W_sqrt)
    eps = 1e-5
    
    I_c_samples = torch.zeros_like(U)
    I_s_samples = torch.zeros_like(U)
    I_c_samples[mask] = I_c(U[mask], V[mask], eps)
    I_s_samples[mask] = I_s(U[mask], V[mask])

    #we need to weight according to the width of the intervals over which the deltas are discretized
    I_c_weighted = I_c_samples * W_mat
    I_s_weighted = I_s_samples * W_mat
    
    #notice that the result are normally sorted from smaller to larger considering their sign
    #we actually want them sorted in terms of their absolute value!
    sigma_c, phi_c = torch.linalg.eigh(I_c_weighted)
    abs_indices = torch.argsort(torch.abs(sigma_c), descending=True)
    sigma_c = sigma_c[abs_indices]
    phi_c_weighted = phi_c[:, abs_indices]

    sigma_s, phi_s_weighted = torch.linalg.eigh(I_s_weighted)
    abs_indices = torch.argsort(torch.abs(sigma_s), descending=True)
    sigma_s = sigma_s[abs_indices]
    phi_s_weighted = phi_s_weighted[:, abs_indices]
    
    phi_c_samples = phi_c_weighted / W_sqrt.unsqueeze(1)
    phi_s_samples = phi_s_weighted / W_sqrt.unsqueeze(1)

    phi_c = build_indicator_evaluator(u, w, phi_c_samples)
    phi_s = build_indicator_evaluator(u, w, phi_s_samples)
    
    return sigma_c, sigma_s, phi_c, phi_s

#N_alpha is the number of modes included in the reconstruction
#very important: Phi_q has to be normalized (not irfftn, but irfftn *dx3)

def compute_grad_V(k, Phi_q, L, u, N_alpha, phi_c, phi_s): 
    device = Phi_q.device
    N = len(Phi_q[:, 0, 0])
    dx3 = (L / N)**3 
    N_modes_s, N_modes_c = N_alpha
    
    #create the 3D grid
    q_whole = 2*torch.pi*torch.fft.fftfreq(N, d=L/N, device=device)
    q_half  = 2*torch.pi*torch.fft.rfftfreq(N, d=L/N, device=device)
    qx, qy, qz = torch.meshgrid(q_whole, q_whole, q_half, indexing='ij')
    q_grid = torch.sqrt(qx**2 + qy**2 + qz**2)
    arg = q_grid / k
    
    mask_q = (q_grid > 0) * (arg < torch.max(u)) * (arg >= torch.min(u))
    phi_s_eval = phi_s(arg)[..., :N_modes_s]
    phi_c_eval = phi_c(arg)[..., :N_modes_c]    

    Phi_q_exp = Phi_q[..., None]
    V_s_q = torch.where(mask_q[..., None], phi_s_eval * Phi_q_exp, 0.0)
    V_c_q = torch.where(mask_q[..., None], phi_c_eval * Phi_q_exp, 0.0)
    
    grad_s = torch.zeros((3, N, N, N, N_modes_s), dtype=torch.float32, device=device)
    grad_c = torch.zeros((3, N, N, N, N_modes_c), dtype=torch.float32, device=device)

    ik = [1j * qx[..., None], 1j * qy[..., None], 1j * qz[..., None]]

    for i in range(3):
        mul_k_s = torch.mul(ik[i], V_s_q).contiguous()
        grad_s[i] = torch.fft.irfftn(mul_k_s, s=(N, N, N), dim=(0, 1, 2)).real / dx3
        
        mul_k_c = torch.mul(ik[i], V_c_q).contiguous()
        grad_c[i] = torch.fft.irfftn(mul_k_c, s=(N, N, N), dim=(0, 1, 2)).real / dx3
        
    return grad_s, grad_c

def compute_envelopes(k, Phi_q, L, u, N_alpha, sigma_c, sigma_s, phi_c, phi_s, threshold):
    device = Phi_q.device 
    N = Phi_q.shape[0]
    dx3 = (L / N)**3 
    #different because the coefficient of the sine needs less modes to be reconstructed
    N_modes_s, N_modes_c = N_alpha 
    
    #prepare the polarization tensors
    k_grid, e_plus, e_cross = compute_polarization_basis(N, L, device, threshold)
    
    #compute the gradients of V_alpha_s, V_alpha_c
    #dimensions = (3, N, N, N, N_modes_alpha)
    grad_s, grad_c = compute_grad_V(k, Phi_q, L, u, N_alpha, phi_c, phi_s)
    
    sigma_s_reduced = sigma_s[:N_modes_s]
    sigma_c_reduced = sigma_c[:N_modes_c]
    
    #this is the tensor source in real space: dimensions are (3, 3, N, N, N, N_modes_alpha)
    #sum over the reconstruction modes (alpha in the paper)
    #after this dimensions are (3, 3, N, N, N)
    tensor_source_A = torch.einsum('axyzm, bxyzm, m -> abxyz', grad_s, grad_s, sigma_s_reduced)
    tensor_source_B = torch.einsum('axyzm, bxyzm, m -> abxyz', grad_c, grad_c, sigma_c_reduced) 
    
    tensor_source_A_k = torch.fft.rfftn(tensor_source_A, dim=(2, 3, 4)) * dx3
    tensor_source_B_k = torch.fft.rfftn(tensor_source_B, dim=(2, 3, 4)) * dx3
    del tensor_source_A, tensor_source_B
    
    A_plus = torch.einsum('abxyz, abxyz->xyz', e_plus, torch.real(tensor_source_A_k)) + 1j*torch.einsum('abxyz, abxyz->xyz', e_plus, torch.imag(tensor_source_A_k))
    B_plus = torch.einsum('abxyz, abxyz->xyz', e_plus, torch.real(tensor_source_B_k)) + 1j*torch.einsum('abxyz, abxyz->xyz', e_plus, torch.imag(tensor_source_B_k))
    A_cross = torch.einsum('abxyz, abxyz->xyz', e_cross, torch.real(tensor_source_A_k)) + 1j*torch.einsum('abxyz, abxyz->xyz', e_cross, torch.imag(tensor_source_A_k))
    B_cross = torch.einsum('abxyz, abxyz->xyz', e_cross, torch.real(tensor_source_B_k)) + 1j*torch.einsum('abxyz, abxyz->xyz', e_cross, torch.imag(tensor_source_B_k))
    
    return -4*k**(-2)*A_plus, 4*k**(-2)*B_plus, -4*k**(-2)*A_cross, 4*k**(-2)*B_cross

#function factory that given the discretization u, the weights w, builds everything

def build_Omega_gw(u, w, threshold=0.5, Delta=1):
    sigma_c, sigma_s, phi_c, phi_s = decompose_kernels(u, w)
    
    def compute_Omega_gw_eval(k, Phi_q, L, N_alpha):
        A_plus, B_plus, A_cross, B_cross = compute_envelopes(
            k, Phi_q, L, u, N_alpha, 
            sigma_c, sigma_s, phi_c, phi_s, 
            threshold
        )
        
        P_A_plus, sigma_A_plus = compute_power_spectrum_k(k, A_plus, L, Delta)
        P_B_plus, sigma_B_plus = compute_power_spectrum_k(k, B_plus, L, Delta)
        P_A_cross, sigma_A_cross = compute_power_spectrum_k(k, A_cross, L, Delta)
        P_B_cross, sigma_B_cross = compute_power_spectrum_k(k, B_cross, L, Delta)
        
        omega = k**3 * (P_A_plus + P_B_plus + P_A_cross + P_B_cross) / (48. * torch.pi**2)
        error = k**3 * (np.sqrt(sigma_A_plus**2 + sigma_B_plus**2 + sigma_A_cross**2 + sigma_B_cross**2)) / (48. * torch.pi**2)
        
        return omega, error

    return compute_Omega_gw_eval#torch.compile(compute_Omega_gw_eval)#
