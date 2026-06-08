import sys, os, time, argparse, traceback
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--fresh', action='store_true')
parser.add_argument('--tend', type=float, default=120.0)
parser.add_argument('--outdir', type=str, default='fipy_marangoni_out')
parser.add_argument('--nxny', type=int, default=160)
parser.add_argument('--ckpt_every', type=int, default=4000)
parser.add_argument('--log_every', type=int, default=500)
parser.add_argument('--sweeps', type=int, default=2)
args = parser.parse_args()

import fipy as fp

OUT = args.outdir
os.makedirs(f'{OUT}/checkpoints', exist_ok=True)
LOG = open(f'{OUT}/sim.log', 'a' if not args.fresh else 'w', buffering=1)

def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')

log('='*70)
log(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] FiPy Marangoni (4-bug-fix v14)')

c_L_eq = 0.02035848
c_S_eq = 0.23825756
A_liq  = 53.305
A_sol  = 5.0 * A_liq

kappa_phi = 4.0
W_dw      = 0.5
L_phi     = 1.0
eta_eq    = float(np.sqrt(kappa_phi / (2.0 * W_dw)))

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
rho_nd    = 1.0
A_drag    = 500.0

nx, ny = args.nxny, args.nxny
dx = 1.0
Lx, Ly = nx*dx, ny*dx

R0 = 15.0
Delta_c = 0.025
c_init = c_L_eq + Delta_c
xc_seed = Lx / 2
yc_seed = Ly * 0.35

dt = 0.5 * min(dx**2/(4*L_phi*kappa_phi),
               dx**2/(4*M_liq*A_liq),
               dx**2/(4*nu_nd))
T_end = args.tend
n_steps = int(T_end / dt)

log(f'  Grid: {nx}×{ny}, T_end*={T_end}, dt={dt:.5f}, n_steps={n_steps}, sweeps={args.sweeps}')
log(f'  Ma={Ma_coeff:.3e}, Sc={nu_nd}, A_drag={A_drag}')
log(f'  c_L_eq={c_L_eq:.5f}, c_S_eq={c_S_eq:.5f}, Δc={Delta_c}, R0={R0}, y_seed={yc_seed:.1f}')

mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dx)
x_cell, y_cell = mesh.cellCenters

phi   = fp.CellVariable(name='phi',   mesh=mesh, value=0.0, hasOld=True)
c     = fp.CellVariable(name='c',     mesh=mesh, value=c_init, hasOld=True)
mu    = fp.CellVariable(name='mu',    mesh=mesh, value=0.0)
omega = fp.CellVariable(name='omega', mesh=mesh, value=0.0, hasOld=True)
psi   = fp.CellVariable(name='psi',   mesh=mesh, value=0.0)

S_mar = fp.CellVariable(name='S_mar', mesh=mesh, value=0.0)
AC_src_exp = fp.CellVariable(name='AC_exp', mesh=mesh, value=0.0)
AC_src_imp = fp.CellVariable(name='AC_imp', mesh=mesh, value=0.0)
mu_src     = fp.CellVariable(name='mu_src', mesh=mesh, value=0.0)

velocity = fp.FaceVariable(mesh=mesh, rank=1, value=0.0)

dist = fp.numerix.sqrt((x_cell - xc_seed)**2 + (y_cell - yc_seed)**2)
phi_init = 0.5 * (1.0 - fp.numerix.tanh((dist - R0) / eta_eq))
phi.setValue(phi_init)
h0 = np.array(phi.value)**3 * (6*np.array(phi.value)**2 - 15*np.array(phi.value) + 10)
c.setValue(h0 * c_S_eq + (1 - h0) * c_init)

all_boundaries = mesh.facesLeft | mesh.facesRight | mesh.facesTop | mesh.facesBottom
c.constrain(c_L_eq, where=mesh.facesBottom)
mu.constrain(0.0, where=mesh.facesBottom)
omega.constrain(0.0, where=all_boundaries)
psi.constrain(0.0, where=all_boundaries)

def solve_kks(c_arr, phi_arr):
    pc = np.clip(phi_arr, 0.0, 1.0)
    h = pc**3 * (6*pc**2 - 15*pc + 10)
    denom = h / A_sol + (1.0 - h) / A_liq
    numer = c_arr - h * c_S_eq - (1.0 - h) * c_L_eq
    mt = numer / denom
    cS = c_S_eq + mt / A_sol
    cL = c_L_eq + mt / A_liq
    return cS, cL, mt, h

def delta_F_kks(cS, cL, mt):
    return 0.5*A_sol*(cS-c_S_eq)**2 - 0.5*A_liq*(cL-c_L_eq)**2 - mt*(cS-cL)

M_c = M_liq * (1.0 - phi) + M_sol * phi

eq_phi = (fp.TransientTerm(coeff=1.0, var=phi) +
          fp.ConvectionTerm(coeff=velocity, var=phi) ==
          fp.DiffusionTerm(coeff=L_phi*kappa_phi, var=phi) +
          fp.ImplicitSourceTerm(coeff=AC_src_imp, var=phi) +
          AC_src_exp)

eq_c  = (fp.TransientTerm(coeff=1.0, var=c) +
         fp.ConvectionTerm(coeff=velocity, var=c) ==
         fp.DiffusionTerm(coeff=M_c, var=mu))

eq_mu = (fp.ImplicitSourceTerm(coeff=1.0, var=mu) == mu_src)

eq_omega = (fp.TransientTerm(coeff=1.0, var=omega) +
            fp.ConvectionTerm(coeff=velocity, var=omega) ==
            fp.DiffusionTerm(coeff=nu_nd, var=omega) +
            S_mar +
            fp.ImplicitSourceTerm(coeff=-A_drag*phi**2, var=omega))

eq_psi = fp.DiffusionTerm(coeff=1.0, var=psi) == -omega

start_step = 0
if not args.fresh:
    ckpts = sorted([f for f in os.listdir(f'{OUT}/checkpoints') if f.startswith('ckpt_')])
    if ckpts:
        data = np.load(f'{OUT}/checkpoints/{ckpts[-1]}')
        if data['phi'].shape != (ny, nx):
            log(f'  *** grid mismatch — refusing resume'); sys.exit(2)
        phi.setValue(data['phi'].flatten()); c.setValue(data['c'].flatten())
        omega.setValue(data['omega'].flatten()); psi.setValue(data['psi'].flatten())
        start_step = int(data['step'])
        log(f'  *** RESUMING at step {start_step} ***')

mass_init = float(fp.numerix.sum(c.value * mesh.cellVolumes))
log(f'  initial mass = {mass_init:.4f}')

xs = (np.arange(nx)+0.5)*dx; ys = (np.arange(ny)+0.5)*dx
X, Y = np.meshgrid(xs, ys)
RR = np.sqrt((X-xc_seed)**2 + (Y-yc_seed)**2)

log('\n' + '='*70)
log(f'{"step":>7} {"t":>8} {"R":>5} {"c_min":>7} {"c_max":>7} {"|ω|max":>9} {"|u|max":>9} {"Δm%":>9} {"wall":>7}')
log('='*70)
t_wall0 = time.time()
hist = {'t': [], 'R': [], 'mass': [], 'u_max': [], 'om_max': []}

try:
    for step in range(start_step, n_steps + 1):
        phi.updateOld(); c.updateOld(); omega.updateOld()

        phi_arr = np.clip(np.array(phi.value), 0, 1)
        c_arr = np.array(c.value)
        cS, cL, mt, h = solve_kks(c_arr, phi_arr)
        delF = delta_F_kks(cS, cL, mt)

        dh = 30 * phi_arr**2 * (1 - phi_arr)**2
        g_p = 2 * phi_arr * (1 - phi_arr) * (1 - 2*phi_arr)
        d2h = 60 * phi_arr * (1 - phi_arr) * (1 - 2*phi_arr)
        g_pp = 2 * (6*phi_arr**2 - 6*phi_arr + 1)
        S_full = -L_phi * (dh * delF + W_dw * g_p)
        dS_dphi = -L_phi * (d2h * delF + W_dw * g_pp)
        dS_stab = np.minimum(dS_dphi, 0.0)
        AC_src_imp.setValue(dS_stab)
        AC_src_exp.setValue(S_full - dS_stab * phi_arr)
        mu_src.setValue(mt)

        grad_c = c.grad
        grad_phi = phi.grad
        cross_z = (np.array(grad_c[0]) * np.array(grad_phi[1]) -
                   np.array(grad_c[1]) * np.array(grad_phi[0]))
        S_mar.setValue((Ma_coeff / rho_nd) * cross_z)

        for _ in range(args.sweeps):
            eq_mu.solve()
            eq_phi.solve(dt=dt)
            eq_c.solve(dt=dt)
            eq_omega.solve(dt=dt)
            eq_psi.solve()

        phi.setValue(np.clip(np.array(phi.value), 0.0, 1.0))

        psi_fg = psi.faceGrad
        velocity.setValue(fp.numerix.array([np.array(psi_fg[1].value),
                                            -np.array(psi_fg[0].value)]))

        if step % args.log_every == 0 or step == n_steps:
            mass = float(fp.numerix.sum(c.value * mesh.cellVolumes))
            derr = abs(mass - mass_init) / mass_init * 100
            phi_2d = np.array(phi.value).reshape((ny, nx))
            c_2d = np.array(c.value).reshape((ny, nx))
            om_max = float(np.abs(np.array(omega.value)).max())
            psi_cg = psi.grad
            vx = np.array(psi_cg[1]); vy = -np.array(psi_cg[0])
            u_max = float(np.sqrt(vx**2 + vy**2).max())
            R_now = float(RR[phi_2d > 0.5].max()) if (phi_2d > 0.5).any() else 0.0
            elapsed = time.time() - t_wall0
            hist['t'].append(step*dt); hist['R'].append(R_now)
            hist['mass'].append(mass); hist['u_max'].append(u_max); hist['om_max'].append(om_max)
            log(f'{step:>7d} {step*dt:>8.2f} {R_now:>5.1f} {c_2d.min():>7.4f} {c_2d.max():>7.4f} '
                f'{om_max:>9.2e} {u_max:>9.2e} {derr:>9.2e} {elapsed:>6.0f}s')

            if not np.isfinite(c.value).all():
                log('*** NaN — aborting ***'); break

        if step > start_step and (step % args.ckpt_every == 0 or step == n_steps):
            phi_2d = np.array(phi.value).reshape((ny, nx))
            c_2d   = np.array(c.value).reshape((ny, nx))
            om_2d  = np.array(omega.value).reshape((ny, nx))
            psi_2d = np.array(psi.value).reshape((ny, nx))
            psi_cg = psi.grad
            vx_2d = np.array(psi_cg[1]).reshape((ny, nx))
            vy_2d = (-np.array(psi_cg[0])).reshape((ny, nx))
            ck = f'{OUT}/checkpoints/ckpt_{step:08d}.npz'
            np.savez_compressed(ck, step=step, phi=phi_2d, c=c_2d,
                                omega=om_2d, psi=psi_2d, ux=vx_2d, uy=vy_2d)
            np.savez(f'{OUT}/history.npz',
                     t=np.array(hist['t']), R=np.array(hist['R']),
                     mass=np.array(hist['mass']),
                     u_max=np.array(hist['u_max']), om_max=np.array(hist['om_max']),
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
log(f'DONE — wrote {OUT}/done.flag')
LOG.close()
