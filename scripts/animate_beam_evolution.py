#!/usr/bin/env python3
"""Complete openPMD series: weighted phase spaces and two RZ density cuts."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from PIL import Image

QE = 1.602176634e-19


def particles(ts, iteration):
    x, y, z, ux, uy, uz, w = [np.asarray(a) for a in ts.get_particle(
        var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'],
        species='electrons', iteration=iteration)]
    valid = np.isfinite(x+y+z+ux+uy+uz+w) & (w > 0)
    x, y, z, ux, uy, uz, w = [a[valid] for a in (x,y,z,ux,uy,uz,w)]
    energy = (np.sqrt(1+ux*ux+uy*uy+uz*uz)-1)*0.51099895
    return x, y, z, ux, uy, uz, w, energy


def density(ts, iteration, theta, n0, zref):
    rho, info = ts.get_field(field='rho_electrons', iteration=iteration,
                             m='all', theta=theta)
    labels = [info.axes[i] for i in range(2)]
    if set(labels) != {'r', 'z'}:
        raise ValueError(f'Unexpected field axes: {labels}')
    image = np.asarray(rho) if labels == ['r', 'z'] else np.asarray(rho).T
    z = np.asarray(info.z)*1e6-zref
    transverse = np.asarray(info.r)*1e6
    return -image/(QE*n0), z, transverse


def gif(paths, target):
    frames = []
    for path in paths:
        with Image.open(path) as source:
            frames.append(source.convert('RGB').quantize(colors=256))
    frames[0].save(target, save_all=True, append_images=frames[1:],
                   duration=180, loop=0, optimize=False)
    for frame in frames:
        frame.close()


def histogram(ax, a, b, charge, edges, norm, title, xlabel, ylabel):
    h, _, _ = np.histogram2d(a, b, bins=edges, weights=charge)
    image = ax.pcolormesh(edges[0], edges[1], np.ma.masked_less_equal(h.T, 0),
                          norm=norm, cmap='magma', shading='flat')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel,
           xlim=(edges[0][0],edges[0][-1]), ylim=(edges[1][0],edges[1][-1]))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    case, out = Path(args.case_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    phase_dir, rho_dir = out/'phase_frames', out/'rho_frames'
    phase_dir.mkdir(); rho_dir.mkdir()
    pt = OpenPMDTimeSeries(str(case/'diags/plasma_electrons'), check_all_files=False)
    ft = OpenPMDTimeSeries(str(case/'diags/fields'), check_all_files=False)
    iterations = sorted(map(int, pt.iterations))
    if set(iterations) != set(map(int, ft.iterations)):
        raise ValueError('Particle and field iterations differ; no frames silently dropped')
    with (case/'guiding_metrics.csv').open(newline='') as stream:
        guiding = {int(r['iteration']):r for r in csv.DictReader(stream)}
    if not set(iterations).issubset(guiding):
        raise ValueError('Missing exact guiding coordinates')
    ptime = dict(zip(map(int, pt.iterations), map(float,pt.t)))
    ftime = dict(zip(map(int, ft.iterations), map(float,ft.t)))
    for it in iterations:
        for time_fs in (ftime[it]*1e15, float(guiding[it]['time_fs'])):
            if not math.isclose(ptime[it]*1e15,time_fs,rel_tol=1e-9,abs_tol=1e-3):
                raise ValueError(f'Inconsistent timestamps at {it}')
    match = re.search(r'^\s*my_constants\.n0\s*=\s*([-+\d.eE]+)',
                      (case/'warpx_used_inputs').read_text(), re.MULTILINE)
    if not match:
        raise ValueError('Missing n0 in archived inputs')
    n0 = float(match.group(1))
    if n0 <= 0:
        raise ValueError('Invalid n0')
    zlo, zhi, xmax, umax, emax, qmax, dmin, dmax = np.inf,-np.inf,1.,1.,50.,1.,0.,1.
    # First pass determines common axes and colour scales over the ENTIRE series.
    for it in iterations:
        zr = float(guiding[it]['z_max_um'])
        x,y,z,ux,uy,uz,w,k = particles(pt,it)
        forward = uz > 0
        if np.any(forward):
            rel = z[forward]*1e6-zr
            zlo, zhi = min(zlo,float(rel.min())), max(zhi,float(rel.max()))
            xmax=max(xmax,float(np.max(np.abs(x[forward])))*1e6)
            umax=max(umax,float(np.max(np.abs(ux[forward]))))
            emax=max(emax,float(k[forward].max()))
            qmax=max(qmax,float(w[forward].sum()*QE*1e12))
        for theta in (0.,np.pi/2):
            d,zz,rr=density(ft,it,theta,n0,zr)
            dmin=min(dmin,float(np.nanmin(d)));dmax=max(dmax,float(np.nanmax(d)))
            zlo=min(zlo,float(zz.min()));zhi=max(zhi,float(zz.max()))
            xmax=max(xmax,float(np.max(np.abs(rr))))
        print(f'[SCALE] iteration={it}',flush=True)
    emax *= 1.03; xmax *= 1.02; umax *= 1.03
    ze=np.linspace(zlo,zhi,181); ke=np.linspace(0,emax,161)
    xe=np.linspace(-xmax,xmax,181); ue=np.linspace(-umax,umax,161)
    norm=LogNorm(vmin=min(1e-4,qmax/1e6),vmax=qmax)
    dnorm=SymLogNorm(linthresh=.05,vmin=dmin,vmax=dmax,base=10)
    metadata={'case_dir':str(case),'n_frames':len(iterations),'iterations':iterations,
              'n0_m3':n0,'phase_selection':'finite positive weights, uz > 0; no lower energy cut',
              'charge_bins':'pC per bin, Cartesian particle projection over all azimuths',
              'rho':'n_e/n0 = -rho_electrons/(e*n0), all saved modes, cuts theta=0 and pi/2',
              'coordinates':'z - exact guiding z_max, micrometres',
              'fixed_scales':{'z_um':[zlo,zhi],'x_um':[-xmax,xmax],
                              'ux':[-umax,umax],'energy_MeV':[0,emax],
                              'density_n0':[dmin,dmax]},'gif_frame_duration_ms':180}
    (out/'visualization_metadata.json').write_text(json.dumps(metadata,indent=2))
    phase_paths=[];rho_paths=[];records=[]
    for it in iterations:
        zr=float(guiding[it]['z_max_um'])
        x,y,z,ux,uy,uz,w,k=particles(pt,it)
        forward=uz>0;hot=forward & (k>=50)
        charge=w*QE*1e12;rel=z*1e6-zr
        title=f'{case.name}\niteration={it} | t={ptime[it]*1e15:.2f} fs | absolute z_frame={zr/1000:.4f} mm'
        fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
        histogram(axes[0],rel[forward],k[forward],charge[forward],(ze,ke),norm,
                  'Forward electrons: longitudinal phase space','z - z_frame (um)','K (MeV)')
        axes[0].axhline(50,color='cyan',lw=1,ls='--')
        histogram(axes[1],x[forward]*1e6,ux[forward],charge[forward],(xe,ue),norm,
                  'Forward electrons: all energies','x (um)','u_x = p_x/(m_e c)')
        im=histogram(axes[2],x[hot]*1e6,ux[hot],charge[hot],(xe,ue),norm,
                     'Forward electrons: K >= 50 MeV','x (um)','u_x = p_x/(m_e c)')
        fig.colorbar(im,ax=list(axes),label='Charge magnitude per bin (pC)')
        fig.suptitle(title,fontsize=11)
        path=phase_dir/f'phase_{it:08d}.png';fig.savefig(path,dpi=100);plt.close(fig);phase_paths.append(path)
        fig,axes=plt.subplots(2,1,figsize=(11,7.3),layout='constrained',sharex=True,sharey=True)
        for ax,theta,plane in zip(axes,(0.,np.pi/2),('x-z','y-z')):
            d,zz,rr=density(ft,it,theta,n0,zr)
            im=ax.pcolormesh(zz,rr,d,norm=dnorm,cmap='RdBu_r',shading='auto',rasterized=True)
            ax.set(title=f'Electron density: {plane} cut, all saved modes',
                   ylabel='Signed transverse coordinate (um)',xlim=(zlo,zhi),ylim=(-xmax,xmax))
        axes[-1].set_xlabel('z - z_frame (um)')
        fig.colorbar(im,ax=list(axes),label='n_e/n0 (signed reconstruction; fixed symlog scale)')
        fig.suptitle(title,fontsize=10)
        path=rho_dir/f'rho_{it:08d}.png';fig.savefig(path,dpi=100);plt.close(fig);rho_paths.append(path)
        records.append({'iteration':it,'time_fs':ptime[it]*1e15,'z_frame_um':zr,
                        'charge_forward_pC':float(charge[forward].sum()),
                        'charge_forward_50MeV_pC':float(charge[hot].sum()),
                        'n_forward_50MeV':int(hot.sum())})
        print(f'[FRAME] iteration={it} Q50={charge[hot].sum():.6g} pC',flush=True)
    with (out/'frame_manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    gif(phase_paths,out/'phase_space_full.gif');gif(rho_paths,out/'rho_electrons_full.gif')
    print(f'[OK] {len(iterations)} frames in each animation: {out}',flush=True)


if __name__ == '__main__':
    main()
