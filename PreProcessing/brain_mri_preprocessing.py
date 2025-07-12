"""
Brain MRI Preprocessing Pipeline
Author: Soorena Salari
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging


def run_command(cmd, **kwargs):
    """Run a shell command and exit on failure."""
    logging.debug(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {' '.join(e.cmd)}")
        sys.exit(e.returncode)


def convert_nii_to_mnc(nifti_path: Path) -> Path:
    mnc_path = nifti_path.with_suffix('.mnc')
    if not mnc_path.exists():
        run_command(['nii2mnc', str(nifti_path), str(mnc_path), '-unsigned'])
    return mnc_path


def n4_bias_correction(input_mnc: Path, output_mnc: Path, mask: Path = None,
                        iterations: str = None, histogram_sharpening: tuple = None,
                        verbose: bool = False):
    cmd = ['N4BiasFieldCorrection', '-d', '3']
    if verbose:
        cmd.append('--verbose')
    if mask:
        cmd += ['-x', str(mask)]
    if iterations:
        cmd += ['-b', '[200]', '-c', iterations]
    if histogram_sharpening:
        cmd += ['--histogram-sharpening', *map(str, histogram_sharpening)]
    cmd += ['-i', str(input_mnc), '-o', str(output_mnc)]
    run_command(cmd)


def skull_strip_and_normalize(nu_mnc: Path, model_dir: Path) -> tuple[Path, Path, Path]:
    head_mni = nu_mnc.with_name(nu_mnc.stem.replace('_nu_1', '_head_mni') + '.mnc')
    tal_xfm  = nu_mnc.with_name(nu_mnc.stem.replace('_nu_1', '_toTal') + '.xfm')
    mask_mni = head_mni.with_name(head_mni.stem.replace('_head_mni', '_mask_mni') + '.mnc')

    if not head_mni.exists() or not tal_xfm.exists():
        run_command([
            'beast_normalize',
            str(nu_mnc), str(head_mni), str(tal_xfm),
            '-modeldir', str(model_dir)
        ])

    if not mask_mni.exists():
        run_command([
            'mincbeast', '-fill', '-median',
            '-conf', str(model_dir / 'default.1mm.conf'),
            str(model_dir), str(head_mni), str(mask_mni)
        ])

    return head_mni, tal_xfm, mask_mni


def resample_mask_to_native(mask_mni: Path, reference_mnc: Path, tal_xfm: Path) -> Path:
    mask_native = mask_mni.with_name(mask_mni.stem.replace('_mask_mni', '_mask_native') + '.mnc')
    if not mask_native.exists():
        run_command([
            'mincresample', str(mask_mni), '-like', str(reference_mnc),
            '-invert_transformation', '-transform', str(tal_xfm),
            str(mask_native), '-short', '-nearest'
        ])
    return mask_native


def apply_skull_mask(mask_native: Path, brain_mnc: Path) -> Path:
    brain_native = brain_mnc.with_name(brain_mnc.stem.replace('.mnc', '_brain_native.mnc'))
    if not brain_native.exists():
        run_command([
            'minccalc', '-expr', 'A[0]>0.5?A[1]:0',
            str(mask_native), str(brain_mnc), str(brain_native),
            '-short', '-unsigned'
        ])
    return brain_native


def linear_registration(input_mnc: Path, template: Path) -> Path:
    stx_xfm = input_mnc.with_name(input_mnc.stem.replace('_nu_2', '_stx') + '.xfm')
    if not stx_xfm.exists():
        run_command([
            'bestlinreg_s', '-lsq12', '-xcorr',
            str(input_mnc), str(template), str(stx_xfm), '-clobber'
        ])
    return stx_xfm


def resample_to_template(nu_mnc: Path, stx_xfm: Path, template: Path) -> Path:
    final_mnc = nu_mnc.with_name(nu_mnc.stem.replace('_nu_2', '_final2') + '.mnc')
    if not final_mnc.exists():
        run_command([
            'mincresample', '-short', '-transform', str(stx_xfm),
            '-like', str(template), str(nu_mnc), str(final_mnc),
            '-trilinear', '-clobber'
        ])
    return final_mnc


def mnc_to_nifti(mnc_path: Path) -> Path:
    nii_path = mnc_path.with_suffix('.nii')
    if not nii_path.exists():
        run_command(['mnc2nii', str(mnc_path), str(nii_path)])
    return nii_path


def process_subject(nifti_path: Path, args: argparse.Namespace):
    logging.info(f"Processing subject: {nifti_path.parent.parent.name}")

    # Step 0: Convert format
    mnc = convert_nii_to_mnc(nifti_path)

    # Step 1: Initial bias correction
    nu1 = mnc.with_name(mnc.stem + '_nu_1.mnc')
    if not nu1.exists():
        n4_bias_correction(mnc, nu1, iterations=f"[{args.iter1}]", verbose=args.verbose)

    # Step 2: Skull-stripping & normalization
    head_mni, tal_xfm, mask_mni = skull_strip_and_normalize(nu1, args.beast_dir)

    # Step 2.2 & 2.3: Mask -> native and remove skull
    mask_native = resample_mask_to_native(mask_mni, mnc, tal_xfm)
    brain_native = apply_skull_mask(mask_native, mnc)

    # Step 3: Refined bias correction
    nu2 = mnc.with_name(mnc.stem + '_nu_2.mnc')
    if not nu2.exists():
        n4_bias_correction(
            brain_native, nu2, mask=mask_native,
            iterations=f"[{args.iter2}]",
            histogram_sharpening=(args.histo_a, args.histo_b, args.histo_c),
            verbose=args.verbose
        )

    # Step 4: Linear registration
    stx = linear_registration(nu2, args.template_brain)
    final = resample_to_template(nu2, stx, args.template_brain)

    # Step 7: Convert back to NIfTI and copy outputs
    nifti_out = mnc_to_nifti(stx)
    dest = args.output_dir / nifti_out.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    nifti_out.replace(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain MRI preprocessing pipeline')
    parser.add_argument('root_dir', type=Path, help='Root directory containing subject folders')
    parser.add_argument('--beast-dir', type=Path, default=Path('beast-library-1.1'), help='BEaST model directory')
    parser.add_argument('--template-brain', type=Path,
                        default=Path('mni_icbm152_nlin_sym_09c_minc2') /
                                'mni_icbm152_t1_tal_nlin_sym_09c-brain.mnc',
                        help='Template brain image')
    parser.add_argument('--output-dir', type=Path, default=Path('output'), help='Directory for processed outputs')
    parser.add_argument('--iter1', type=str, default='200x200x200x200x0', help='Iterations for initial N4 correction')
    parser.add_argument('--iter2', type=str, default='300x300x300x200x0', help='Iterations for refined N4 correction')
    parser.add_argument('--histo-a', type=float, default=0.05, help='Histogram sharpening A')
    parser.add_argument('--histo-b', type=float, default=0.01, help='Histogram sharpening B')
    parser.add_argument('--histo-c', type=int, default=1000, help='Histogram sharpening C')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    subjects = sorted(p for p in args.root_dir.iterdir() if p.is_dir() and p.name.startswith('sub-'))
    for subj in subjects:
        nifti_file = subj / 'anat' / f"{subj.name}_acq-MP2RAGE_run-01_T1w.nii"
        if nifti_file.exists():
            process_subject(nifti_file, args)
        else:
            logging.warning(f"Missing file for {subj.name}: {nifti_file}")
