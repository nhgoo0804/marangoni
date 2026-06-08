import sys, os, time, argparse, traceback
import numpy as np
from scipy.fft import dstn, idstn

parser = argparse.ArgumentParser()
parser.add_argument('--fresh', action='store_true', help='Ignore checkpoints')
parser.add_argument('--tend', type=float, default=120.0, help='Final dimensionless time')
parser.add_argument('--outdir', type=str, default='marangoni_out', help='Output directory')
parser.add_argument('--nxny', type=int, default=160, help='Grid size (square)')
parser.add_argument('--ckpt_every', type=int, default=4000, help='Steps between checkpoints')
parser.add_argument('--log_every', type=int, default=500, help='Steps between log lines')
args = parser.parse_args()

OUT = args.outdir
os.makedirs(f'{OUT}/checkpoints', exist_ok=True)
LOG = open(f'{OUT}/sim.log', 'a' if not args.fresh else 'w', buffering=1)

def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')

log('='*70)
log(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Marangoni simulation start')
log(f'  Grid: {args.nxny}×{args.nxny}, T_end* = {args.tend}')
log(f'  fresh={args.fresh}, outdir={OUT}, ckpt_every={args.ckpt_every}')

c_L_eq = 0.02035848
c_S_eq = 0.23825756
A_liq  = 53.305
A_sol  = 5.0 * A_liq

kappa_phi = 4.0
W_dw      = 0.5
L_phi     = 1.0
eta_eq    = np.sqrt(kappa_phi / (2.0 * W_dw))

D_liq_eff = 1.0
D_sol_eff = 2.0e-5
M_liq = D_liq_eff / A_liq
M_sol = D_sol_eff / A_liq

dgamma_dc = 20.0
L0_phys   = 1.0e-7
D_phys    = 5.0e-9
nu_phys   = 5.0e-7
rho_phys  = 2400.0
Ma_coeff  = dgamma_dc * L0_phys / (rho_phys * D_phys * nu_phys)
nu_nd     = nu_phys / D_phys
A_drag    = 500.0

nx, ny = args.nxny, args.nxny
dx = 1.0
Lx, Ly = nx*dx, ny*dx

R0 = 15.0
Delta_c = 0.025
c_init  = c_L_eq + Delta_c
xc_seed = Lx / 2
yc_seed = Ly * 0.35

dt_AC      = dx**2 / (4.0 * L_phi * kappa_phi)
dt_CH      = dx**2 / (4.0 * M_liq * A_liq)
dt_visc    = dx**2 / (4.0 * nu_nd)
dt = 0.5 * min(dt_AC, dt_CH, dt_visc)
T_end = args.tend
n_steps = int(T_end / dt)

log(f'  c_L_eq={c_L_eq:.5f}, c_S_eq={c_S_eq:.5f}, Δc_init={Delta_c}')
log(f'  A_liq={A_liq:.2f}, A_sol={A_sol:.2f}, κ_φ={kappa_phi}, W_dw={W_dw}, L_φ={L_phi}')
log(f'  Ma={Ma_coeff:.3e}, Sc={nu_nd}, A_drag={A_drag}')
log(f'  R0={R0}, particle at y={yc_seed:.1f} ({yc_seed/Ly*100:.0f}%)')
log(f'  dt={dt:.5f}, n_steps={n_steps}, ckpt_every={args.ckpt_every}')

def solve_kks(c_arr, phi_arr):
    pc = np.clip(phi_arr, 0.0, 1.0)
    h = pc**3 * (6*pc**2 - 15*pc + 10)
    denom = h / A_sol + (1.0 - h) / A_liq
    numer = c_arr - h * c_S_eq - (1.0 - h) * c_L_eq
    mu = numer / denom
    cS = c_S_eq + mu / A_sol
    cL = c_L_eq + mu / A_liq
    return cS, cL, mu, h

def delta_F_kks(cS, cL, mu):
    return 0.5*A_sol*(cS-c_S_eq)**2 - 0.5*A_liq*(cL-c_L_eq)**2 - mu*(cS-cL)

def laplacian(F, mode='edge'):
    Fp = np.pad(F, 1, mode=mode if mode != 'zero' else 'constant')
    return (Fp[2:,1:-1] + Fp[:-2,1:-1] + Fp[1:-1,2:] + Fp[1:-1,:-2] - 4*Fp[1:-1,1:-1]) / dx**2

def grad_centered(F, mode='edge'):
    Fp = np.pad(F, 1, mode=mode if mode != 'zero' else 'constant')
    dFdx = (Fp[1:-1, 2:] - Fp[1:-1, :-2]) / (2*dx)
    dFdy = (Fp[2:, 1:-1] - Fp[:-2, 1:-1]) / (2*dx)
    return dFdx, dFdy

def ch_rhs(mu, M_cell, c, ux, uy):
    Mp = np.pad(M_cell, 1, mode='edge')
    Mx = 0.5 * (Mp[1:-1, 1:] + Mp[1:-1, :-1])
    My = 0.5 * (Mp[1:, 1:-1] + Mp[:-1, 1:-1])
    mu_p = np.pad(mu, 1, mode='edge')
    mu_p[0, 1:-1] = -mu[0, :]
    Jx = Mx * (mu_p[1:-1, 1:] - mu_p[1:-1, :-1]) / dx
    Jy = My * (mu_p[1:, 1:-1] - mu_p[:-1, 1:-1]) / dx
    Jx[:, 0] = 0; Jx[:, -1] = 0; Jy[-1, :] = 0
    div = (Jx[:, 1:] - Jx[:, :-1]) / dx + (Jy[1:, :] - Jy[:-1, :]) / dx
    c_p = np.pad(c, 1, mode='edge')
    c_p[0, 1:-1] = c_L_eq
    dcdx = (c_p[1:-1, 2:] - c_p[1:-1, :-2]) / (2*dx)
    dcdy = (c_p[2:, 1:-1] - c_p[:-2, 1:-1]) / (2*dx)
    adv = -(ux * dcdx + uy * dcdy)
    return div + adv

i_idx = np.arange(nx)
j_idx = np.arange(ny)
kx_eig = (2.0/dx**2) * (1 - np.cos(np.pi * (i_idx + 0.5) / nx))
ky_eig = (2.0/dx**2) * (1 - np.cos(np.pi * (j_idx + 0.5) / ny))
eig_lap = -(kx_eig[None, :] + ky_eig[:, None])
eig_lap[eig_lap == 0] = 1.0

def poisson_dst(omega):
    rhs = -omega
    rhs_hat = dstn(rhs, type=2, norm='ortho')
    psi_hat = rhs_hat / eig_lap
    return idstn(psi_hat, type=2, norm='ortho')

def velocity_from_psi(psi):
    psi_p = np.pad(psi, 1, mode='constant')
    ux = (psi_p[2:, 1:-1] - psi_p[:-2, 1:-1]) / (2*dx)
    uy = -(psi_p[1:-1, 2:] - psi_p[1:-1, :-2]) / (2*dx)
    return ux, uy

xs = (np.arange(nx) + 0.5) * dx
ys = (np.arange(ny) + 0.5) * dx
X, Y = np.meshgrid(xs, ys)
RR = np.sqrt((X - xc_seed)**2 + (Y - yc_seed)**2)

start_step = 0
if not args.fresh:
    ckpts = sorted([f for f in os.listdir(f'{OUT}/checkpoints') if f.startswith('ckpt_')])
    if ckpts:
        latest = ckpts[-1]
        data = np.load(f'{OUT}/checkpoints/{latest}')
        if data['phi'].shape != (ny, nx):
            log(f'  *** Checkpoint grid {data["phi"].shape} ≠ requested ({ny},{nx}). ')
            log(f'  *** Refusing to resume across grids. Use --fresh or different --outdir.')
            sys.exit(2)
        phi   = data['phi']; c = data['c']
        omega = data['omega']; psi = data['psi']
        ux    = data['ux']; uy = data['uy']
        start_step = int(data['step'])
        log(f'  *** RESUMING from {latest} at step {start_step} ***')

if start_step == 0:
    phi = 0.5 * (1.0 - np.tanh((RR - R0) / eta_eq))
    h_init = phi**3 * (6*phi**2 - 15*phi + 10)
    c = h_init * c_S_eq + (1.0 - h_init) * c_init
    omega = np.zeros_like(phi)
    psi   = np.zeros_like(phi)
    ux    = np.zeros_like(phi)
    uy    = np.zeros_like(phi)

mass_init = np.sum(c) * dx * dx
log(f'  initial mass = {mass_init:.4f}')

log('\n' + '='*70)
log(f'{"step":>7} {"t":>8} {"R":>5} {"c_min":>7} {"c_max":>7} {"|ω|max":>9} {"|u|max":>9} {"Δm%":>9} {"wall":>7}')
log('='*70)

t_wall0 = time.time()
hist = {'t': [], 'R': [], 'mass': [], 'u_max': [], 'om_max': []}

try:
    for step in range(start_step, n_steps + 1):
        c[0, :] = c_L_eq

        cS, cL, mu, h = solve_kks(c, phi)
        delF = delta_F_kks(cS, cL, mu)

        pc = np.clip(phi, 0, 1)
        dh = 30 * pc**2 * (1 - pc)**2
        g_p = 2 * pc * (1 - pc) * (1 - 2*pc)
        dphidx, dphidy = grad_centered(phi)
        adv_phi = -(ux * dphidx + uy * dphidy)
        rhs_phi = L_phi * (kappa_phi * laplacian(phi) - W_dw * g_p - dh * delF) + adv_phi

        M_cell = M_liq * (1 - h) + M_sol * h
        rhs_c = ch_rhs(mu, M_cell, c, ux, uy)

        dcdx, dcdy = grad_centered(c)
        dphidx, dphidy = grad_centered(phi)
        cross_z = dcdx * dphidy - dcdy * dphidx
        S_mar = Ma_coeff * cross_z
        dom_dx, dom_dy = grad_centered(omega, mode='zero')
        adv_om = -(ux * dom_dx + uy * dom_dy)
        rhs_om = nu_nd * laplacian(omega, mode='zero') + S_mar + adv_om

        phi = np.clip(phi + dt * rhs_phi, 0, 1)
        c   = c + dt * rhs_c
        omega = (omega + dt * rhs_om) / (1.0 + dt * A_drag * pc**2)

        c[0, :] = c_L_eq
        omega[0, :] = 0; omega[-1, :] = 0; omega[:, 0] = 0; omega[:, -1] = 0

        psi = poisson_dst(omega)
        ux, uy = velocity_from_psi(psi)

        if step % args.log_every == 0 or step == n_steps:
            mass = np.sum(c) * dx * dx
            derr = abs(mass - mass_init) / mass_init * 100
            u_max = float(np.sqrt(ux**2 + uy**2).max())
            om_max = float(np.abs(omega).max())
            R_now = float(RR[phi > 0.5].max()) if (phi > 0.5).any() else 0.0
            elapsed = time.time() - t_wall0
            hist['t'].append(step*dt); hist['R'].append(R_now)
            hist['mass'].append(mass); hist['u_max'].append(u_max); hist['om_max'].append(om_max)
            log(f'{step:>7d} {step*dt:>8.2f} {R_now:>5.1f} {c.min():>7.4f} {c.max():>7.4f} '
                f'{om_max:>9.2e} {u_max:>9.2e} {derr:>9.2e} {elapsed:>6.0f}s')

            if not (np.isfinite(c).all() and np.isfinite(phi).all() and np.isfinite(omega).all()):
                log('*** NaN detected — aborting ***')
                break

        if step > start_step and (step % args.ckpt_every == 0 or step == n_steps):
            ck = f'{OUT}/checkpoints/ckpt_{step:08d}.npz'
            np.savez_compressed(ck, step=step, phi=phi, c=c,
                                omega=omega, psi=psi, ux=ux, uy=uy)
            np.savez(f'{OUT}/history.npz',
                     t=np.array(hist['t']), R=np.array(hist['R']),
                     mass=np.array(hist['mass']), u_max=np.array(hist['u_max']),
                     om_max=np.array(hist['om_max']),
                     params=dict(nx=nx, ny=ny, dx=dx, dt=dt, T_end=T_end,
                                 c_L_eq=c_L_eq, c_S_eq=c_S_eq, R0=R0,
                                 yc_seed=yc_seed, Ma=Ma_coeff, Sc=nu_nd))

except Exception:
    log('\n!!! EXCEPTION !!!')
    log(traceback.format_exc())
    raise

wall = time.time() - t_wall0
log(f'\nWall time: {wall:.1f} s')
log(f'Steps/sec: {(n_steps - start_step)/max(wall, 1e-3):.0f}')

with open(f'{OUT}/done.flag', 'w') as f:
    f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f't_end_reached={n_steps*dt}\nwall_time_sec={wall:.1f}\n')
log(f'\nDONE — wrote {OUT}/done.flag')
LOG.close()
