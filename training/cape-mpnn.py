#!/usr/bin/env python

import os
import sys
import glob
import traceback
import argparse
import shutil
import warnings
import copy
from tqdm.auto import tqdm

import numpy as np

import torch

import kit
import kit.globals as G
from kit.log import log_info, log_debug
from kit.data import DD, Split, str_to_file
from kit.path import join
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.bioinf.immuno.mhc_2 import Mhc2PredictorPwm
from kit.hyp import get_hyp_params

from CAPE.MPNN.utils import ModelManager
from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.data.utils import get_data_loaders_pdb, shutdown, \
    get_structure_data_sets, \
    get_preference_pairs, get_data_loader_preferences, get_avg_preference_score
from CAPE.MPNN.ProteinMPNN.training.utils import StructureLoader


warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True", category=UserWarning)

TQDM = False
# NOTE on 'seed' (added for replicate runs): it is part of the model hash so that two
# runs differing only in seed land in DIFFERENT model directories instead of colliding
# (and tripping the overwrite guard). This is backward-compatible: --seed defaults to
# None, and ModelManager's d_for_hash keeps only non-None hparams
# (libs/CAPE/MPNN/utils.py), so every existing seed-less model keeps its current id.
# Verified: f16b51e6, 7e908919, 54e1642c etc. are unaffected.
ModelManager.dpo_hparams_names = ['mhc_1_alleles', 'base_model_name', 'beta', 'reload_data_every_n_epochs',
                    'temperature', 'lr', 'num_examples_per_epoch', 'batch_size',
                    'max_protein_length', 'mhc_1_predictor', 'rescut', 'proteome_file_name',
                    'preference_sampling_method', 'mhc_2_predictor', 'mhc_2_alleles',
                    'charge_reg_weight', 'charge_reg_target', 'seed']
hparams_fixed = DD.from_dict({
    k: None for k in [
        # due to data loading
        'reload_data_every_n_epochs',
        'num_examples_per_epoch',
        'max_protein_length',
        'rescut',
        'num_epochs',
        # no technical reason
        'mhc_1_predictor',
        'mhc_1_alleles',
        'mhc_2_predictor',
        'mhc_2_alleles']
})


def loop_over_data_loader(model, split):
    preference_data_loaders = model.dpo_preference_data_loaders
    optimizer = model.dpo_optimizer
    beta = model.dpo_hparams.beta
    temperature = model.dpo_hparams.temperature

    model.torch.to(kit.DEVICE)

    if split == Split.TRAIN:
        model.torch.train()
        torch.set_grad_enabled(True)
    else:
        model.torch.eval()
        torch.set_grad_enabled(False)

    loss_loop = []

    iterator = tqdm(preference_data_loaders[split]) if TQDM else preference_data_loaders[split]
    for preference_batch in iterator:
        max_length, structure_batch_list, sample_dict_w, base_log_probs_w, sample_dict_l, base_log_probs_l, examples = \
            preference_batch
    
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
                    visible_list_list, masked_list_list, masked_chain_length_list_list, \
                    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
                    tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
                    bias_by_res_all, tied_beta = structure_batch_list

        B = len(X)
        decoding_order = torch.arange(max_length).repeat(B, 1).to(kit.DEVICE)

        S_w = sample_dict_w['S']  # shape = (B, L)
        S_l = sample_dict_l['S']

        if split == Split.TRAIN:
            optimizer.zero_grad()
        
        model_result_w = model.torch(X, S_w, mask, chain_M, residue_idx, chain_encoding_all, decoding_order,
                omit_AA_mask=omit_AA_mask, temperature=temperature)  # shape = (B, L, 21)

        model_result_l = model.torch(X, S_l, mask, chain_M, residue_idx, chain_encoding_all, decoding_order,
                omit_AA_mask=omit_AA_mask, temperature=temperature)  # shape = (B, L, 21)

        log_pi_theta_w = model_result_w.gather(-1, S_w.unsqueeze(-1)).squeeze(-1).sum(1)  # shape = (B,)
        log_pi_theta_l = model_result_l.gather(-1, S_l.unsqueeze(-1)).squeeze(-1).sum(1)  # shape = (B,)
        
        log_pi_base_w = base_log_probs_w.gather(-1, S_w.unsqueeze(-1)).squeeze(-1).sum(1)  # shape = (B,)
        log_pi_base_l = base_log_probs_l.gather(-1, S_l.unsqueeze(-1)).squeeze(-1).sum(1)  # shape = (B,)
        
        loss = -torch.log(torch.sigmoid(beta * (log_pi_theta_w - log_pi_base_w - log_pi_theta_l + log_pi_base_l) ))  # shape = (B,)

        loss_avg = loss.mean()  # shape = (1,)

        if split == Split.TRAIN:
            loss_avg.backward()
            optimizer.step()
            model.dpo_tuning_steps += 1
        elif model.dpo_tuning_steps == 0:
            assert float((log_pi_theta_w.exp() - log_pi_base_w.exp()).abs().max()) < 0.01
            assert float((log_pi_theta_l.exp() - log_pi_base_l.exp()).abs().max()) < 0.01

        loss_loop.append(loss_avg.item())

    loss_loop = np.average(loss_loop)
    model.dpo_losses[split].append(loss_loop)

    model.torch.to("cpu")


def save_model(model_cape, epoch, reload):
    losses_train = model_cape.dpo_losses[Split.TRAIN]
    losses_val = model_cape.dpo_losses[Split.VAL]

    log_info(f"{model_cape.id} Epoch {epoch+1} - "
                f"Mean train loss: {losses_train[-1]:.3f}, "
                f"Mean val loss: {losses_val[-1]:.3f}"
    )
    str_to_file(
        f"{reload}\t{epoch+1}\t"
        f"{losses_train[-1]}\t"
        f"{losses_val[-1]}\n", 
        model_cape.get_losses_file_path(), append=True
    )

    # save latest model
    ckpt_last_path = join(model_cape.path, "ckpts", f"last.pt")
    str_to_file(
        G.JOB.ID,
        join(model_cape.path, "jobs", "ckpts", f"last.txt"), 
        append=False
    )
    model_cape.torch.save_to_file(ckpt_last_path)

    # save model every n epochs
    if (epoch + 1) % args.save_model_every_n_epochs == 0:
        shutil.copyfile(
            ckpt_last_path,
            os.path.join(model_cape.path, "ckpts", f"epoch_{epoch+1}.pt")
        )
        str_to_file(
            G.JOB.ID,
            join(model_cape.path, "jobs", "ckpts", f"epoch_{epoch+1}.txt"), 
            append=False
        )

    # it the current model is the best, save it
    if len(losses_val) > 1 and losses_val[-1] < np.min(losses_val[:-1]):
        shutil.copyfile(
            ckpt_last_path, 
            os.path.join(model_cape.path, "ckpts", f"best.pt")
        )


def main(args, _):
    # Seed every RNG stream that affects training when --seed is given. Without this the
    # flag would only relabel the run (changing its hash) while leaving it as nondeterministic
    # as before, which would NOT be a controlled replicate. Training is unseeded by default,
    # matching how all previously deposited models were trained: the PDB loader shuffles and
    # samples a random chain/assembly per cluster, so seed-less runs stay nondeterministic.
    if getattr(args, "seed", None) is not None:
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        log_info(f"Seeded RNGs with --seed {args.seed} (run is a controlled replicate; "
                 f"seed is part of the model hash, so this run gets its own model directory).")
    else:
        log_info("No --seed given: training is unseeded (historical default).")

    start_epoch = 0
    splits = [Split.TRAIN, Split.VAL]
    # set paths
    CapeMPNN.base_model_pt_dir_path = os.path.join(
        G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, 'CAPE-MPNN', 'base_hparams')
    # Use a local (node-local NVMe/ext4) cache if available, to avoid slow NFS reads
    # during per-epoch dataset reloads. PDB_CACHE_DIR overrides the path (e.g. an sbatch
    # stages the dataset to $TMPDIR on the compute node and exports it); if unset we keep
    # the original validated local-machine path, then fall back to the network mount.
    _local_pdb_cache = os.environ.get(
        "PDB_CACHE_DIR", os.environ.get("PDB_CACHE_DIR", ""))
    if os.path.isdir(_local_pdb_cache):
        data_path = _local_pdb_cache
        log_info(f"Using local PDB cache: {data_path}")
    else:
        data_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "pdb_2021aug02")
        log_info(f"Local PDB cache not found, using network mount: {data_path}")

    if args.hyp is not None:  # are hyperparameters given? (single or multiple models)
        for _ in range(args.hyp_n):
            ModelManager(get_hyp_params(args.hyp))
    else:  # single run
        if args.cont_ckpt_id is not None:  # continue training
            log_info(f"Continue training {args.cont_ckpt_id}")
            _tmp = args.cont_ckpt_id.split("@")
            _m = ModelManager(load_ckpt_id=_tmp[0])
            start_epoch = _m.last_dpo_epoch
            if len(_tmp) > 1:
                assert len(_tmp) == 2
                _tmp = eval(_tmp[1])
                for k, v in _tmp.items():
                    assert k in ['num_epochs']
                    _m.dpo_hparams[k] = v
        else:
            ModelManager(DD.from_dict(vars(args)))

    cape_model_ids = list(ModelManager.models.keys())
    
    log_info(f"augment_eps == 0.0... to avoid differences betweeen the base and the CAPE model")

    # load models
    for model_id in cape_model_ids:
        model = ModelManager.models[model_id]

        for k, v in hparams_fixed.items():
            if v is not None and v != model.dpo_hparams[k]:
                raise ValueError(f"Hyperparameter {k} is fixed to {v} but is {model.dpo_hparams[k]}")
            hparams_fixed[k] = model.dpo_hparams[k]

        if model.dpo_hparams.base_model_name not in ModelManager.models:
            log_info(f"Load base model")
            model_base = CapeMPNN.from_file(model.dpo_hparams.base_model_name)
            model_base.eval()
            model_base.to("cpu")
            ModelManager(None, model.dpo_hparams.base_model_name).torch = model_base
        else:
            model_base = ModelManager.models[model.dpo_hparams.base_model_name].torch
        
        if model.torch is None:
            log_info(f"Copy base model")
            model_cape = copy.deepcopy(model_base)
            model_cape.eval()
            model_cape.to("cpu")
            model.torch = model_cape
        else:
            model_cape = model.torch

        model.dpo_optimizer = torch.optim.Adam(
            model_cape.parameters(), 
            lr=model.dpo_hparams.lr, 
            weight_decay=0
        )

        str_to_file("reload\tepoch\ttrain\tval\n", 
                    model.get_preference_scores_file_path(), 
                    append=True
        )
        str_to_file("reload\tepoch\ttrain\tval\n", 
                    model.get_losses_file_path(), 
                    append=True
        )

    # MHC-I scoring is optional (mirrors the MHC-II guard below). Pass
    # --mhc_1_predictor none to disable it — e.g. for the MHC-II-only ablation,
    # which attributes the MHC-II reduction to the MHC-II
    # objective alone. 'none' is a non-None value and stays in the model hash
    # (distinct from 'pwm'), so a disabled-MHC-I run gets its own model directory
    # and cannot collide with the dual-objective production weights (057975d5).
    immuno = {}
    _mhc1_pred_name = hparams_fixed.mhc_1_predictor
    if _mhc1_pred_name is not None and str(_mhc1_pred_name).lower() != 'none':
        mhc_1_predictor = DD.from_yaml(
            os.path.join(
                G.PROJECT_ENV.CONFIG,
                'immuno',
                'mhc_1_predictor',
                f'{_mhc1_pred_name}.yaml'
            )
        )

        mhc_1_alleles = hparams_fixed.mhc_1_alleles.split('+')

        mhc_1_pred_args = {
            'data_dir_path': mhc_1_predictor.PREDICTOR_MHC_I.FOLDER,
            'limit': mhc_1_predictor.PREDICTOR_MHC_I.LIMIT,
        }
        if "LIMIT_CALIBRATION" in mhc_1_predictor.PREDICTOR_MHC_I:
            mhc_1_pred_args['limit_calibration'] = mhc_1_predictor.PREDICTOR_MHC_I.LIMIT_CALIBRATION
        predictor_MHC_I = Mhc1Predictor.get_predictor(mhc_1_predictor.PREDICTOR_MHC_I.NAME)(**mhc_1_pred_args)
        immuno["MHC_I"] = (predictor_MHC_I, mhc_1_alleles, mhc_1_predictor.PREDICTOR_MHC_I.LENGTHS)
        log_info(f"MHC-I predictor enabled: {_mhc1_pred_name}, alleles={hparams_fixed.mhc_1_alleles}")
    else:
        log_info("MHC-I predictor DISABLED (--mhc_1_predictor none): MHC-II-only run")

    if hparams_fixed.mhc_2_predictor is not None:
        mhc_2_cfg = DD.from_yaml(
            os.path.join(
                G.PROJECT_ENV.CONFIG,
                'immuno',
                'mhc_2_predictor',
                f'{hparams_fixed.mhc_2_predictor}.yaml'
            )
        )
        mhc_2_alleles = hparams_fixed.mhc_2_alleles
        predictor_MHC_II = Mhc2PredictorPwm(
            data_dir_path=mhc_2_cfg.PREDICTOR_MHC_II.FOLDER,
            limit=float(mhc_2_cfg.PREDICTOR_MHC_II.LIMIT),
        )
        immuno['MHC_II'] = (predictor_MHC_II, mhc_2_alleles, mhc_2_cfg.PREDICTOR_MHC_II.LENGTHS)
        log_info(f"MHC-II predictor enabled: {hparams_fixed.mhc_2_predictor}, alleles={mhc_2_alleles}")

    # Acidic-composition regulariser (curbs the "glutamate crutch").
    # Read from args so it applies to fresh runs; only active when weight > 0.
    if getattr(args, "charge_reg_weight", 0.0) and args.charge_reg_weight > 0:
        immuno['CHARGE_REG'] = {"weight": args.charge_reg_weight, "target": args.charge_reg_target}
        log_info(f"Charge regulariser ON: weight={args.charge_reg_weight}, "
                 f"acidic target={args.charge_reg_target}")

    # At least one immunogenicity objective must be active, otherwise the DPO
    # preference signal is empty and the run is meaningless.
    assert ('MHC_I' in immuno) or ('MHC_II' in immuno), \
        "No immunogenicity objective active: enable at least one of MHC-I / MHC-II."

    # Hard safety guard; project rule: never overwrite weights. A fresh run
    # (no --cont_ckpt_id) must target an EMPTY ckpts/ directory. If any checkpoint
    # already exists there, abort rather than silently overwrite it epoch-by-epoch
    # (the model-hash-collision trap that lost the v1 de-immunised weights).
    if args.cont_ckpt_id is None:
        for _mid in list(ModelManager.models.keys()):
            _m = ModelManager.models[_mid]
            if getattr(_m, 'is_base_model', False):
                continue
            _ckpt_dir = os.path.join(_m.path, "ckpts")
            _existing = sorted(glob.glob(os.path.join(_ckpt_dir, "*.pt"))) if os.path.isdir(_ckpt_dir) else []
            if _existing:
                raise RuntimeError(
                    f"REFUSING TO OVERWRITE existing weights: {_ckpt_dir} already contains "
                    f"{len(_existing)} checkpoint(s) [{', '.join(os.path.basename(p) for p in _existing[:6])}]. "
                    f"This fresh run (model id {_mid}) would overwrite them. Back up/rename the "
                    f"directory first, or resume with --cont_ckpt_id."
                )
            log_info(f"Safety check OK: {_ckpt_dir} is empty; fresh run may proceed (model id {_mid}).")

    data_loaders_pdb = get_data_loaders_pdb(
        data_path, splits, hparams_fixed.rescut, args.debug)

    log_info(f"Starting training...")

    reload = 0
    for epoch in range(start_epoch, hparams_fixed.num_epochs):
        log_info(f"Start epoch {epoch}")
        if epoch % hparams_fixed.reload_data_every_n_epochs == 0:
            log_debug("  Call: get_structure_data_sets")
            structure_data_sets = get_structure_data_sets(
                data_loaders_pdb, 
                hparams_fixed.max_protein_length, 
                hparams_fixed.num_examples_per_epoch, 
                args.multithreading
            )

            reload += 1
            for model_id in cape_model_ids:
                model_cape = ModelManager.models[model_id]
                model_base = ModelManager.models[model_cape.dpo_hparams.base_model_name]

                log_info(f"  {model_id} Start reload structure_data_loaders")
                model_cape.dpo_structure_data_loaders = {
                    split: StructureLoader(
                            structure_data_sets[split], 
                            batch_size=model_cape.dpo_hparams.batch_size
                            ) for split in splits
                }

                text = ""
                avg_preference_scores = {}
                with torch.no_grad():
                    for split in splits:
                        log_debug(f"  Split: {split}")
                        log_debug(f"   Call: get_preference_pairs {model_cape.dpo_hparams.preference_sampling_method}")
                        preference_pairs = get_preference_pairs(
                            model_cape,
                            immuno,
                            split,
                            preference_sampling_method=model_cape.dpo_hparams.preference_sampling_method
                        )
                        
                        log_debug("   Call: get_data_loader_preferences")
                        model_cape.dpo_preference_data_loaders[split] = get_data_loader_preferences(
                            preference_pairs, 
                            model.dpo_hparams.batch_size
                        )
                        
                        log_debug("   Call: get_avg_preference_score")
                        avg_preference_scores[split] = get_avg_preference_score(preference_pairs)
                        text += f" {split}: {avg_preference_scores[split]:.1f}"

                log_debug("   Call: loop_over_data_loader")
                loop_over_data_loader(model_cape, Split.VAL)
                losses_val = model_cape.dpo_losses[Split.VAL]

                log_info(f"{model_id}  Mean preference score: {text}, Mean val loss: {losses_val[-1]:.3f}")

                str_to_file(
                    f"{reload}\t{epoch}\t"
                    f"{avg_preference_scores[Split.TRAIN]}\t"
                    f"{avg_preference_scores[Split.VAL]}\n", 
                    model_cape.get_preference_scores_file_path(), append=True
                )
                str_to_file(
                    f"{reload}\t{epoch}\t"
                    "\t"
                    f"{losses_val[-1]}\n", 
                    model_cape.get_losses_file_path(), append=True
                )
            
        for model_id in cape_model_ids:
            model_cape = ModelManager.models[model_id]

            for split in splits:
                loop_over_data_loader(model_cape, split)

            save_model(model_cape, epoch, reload)

        if len(cape_model_ids) == 1 and stop_early(model_cape.dpo_losses):
            break

    if args.multithreading:
        shutdown()


def stop_early(losses):
    return False


if __name__ == "__main__":
    arg_string = None
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # for training
    argparser.add_argument("--mhc_1_alleles", type=str, default="HLA-A*02:01",
        help="alleles to deimmunize (e.g. HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01)")
    argparser.add_argument("--base_model_name", type=str, default="v_48_020", help="name of the base model")
    argparser.add_argument("--beta", type=float, default=0.05, help="weight of the KL term in the implicit RLHF loss")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=5, help="reload training data every n epochs")
    argparser.add_argument("--temperature", type=float, default=0.2, help="the temperature to use for sampling")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--lr", type=float, default=3e-7, help="learning rate for fine-tuning the model")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=10000, help="number of examples per epoch")
    argparser.add_argument("--batch_size", type=int, default=1200, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=500, help="maximum length of the protein complex")
    argparser.add_argument("--mhc_1_predictor", type=str, default='pwm', help='the yaml file (in configs/CAPE/immuno/mhc_1_predictor) defining the MHC-I predictor to use')
    argparser.add_argument("--mhc_2_predictor", type=str, default=None, help='the yaml file (in configs/CAPE/immuno/mhc_2_predictor) defining the MHC-II predictor; omit to disable MHC-II scoring')
    argparser.add_argument("--mhc_2_alleles", type=str, default="DRB1_0101+DRB1_0301+DRB1_0401+DRB1_0701+DRB1_1101+DRB1_1501", help='MHC-II alleles to score against (DRB1_XXXX format, "+"-separated)')
    argparser.add_argument("--charge_reg_weight", type=float, default=0.0, help='weight (lambda) of the acidic-composition regulariser in the DPO reward; 0 disables it. Penalises acidic (D+E) content above --charge_reg_target to curb the "glutamate crutch". Try 0.25-1.0.')
    argparser.add_argument("--charge_reg_target", type=float, default=0.15, help='natural acidic (D+E) fraction; only acidic content ABOVE this is penalised')
    argparser.add_argument("--proteome_file_name", type=str, help='the fasta file name including the proteome which will not be immunogenic (in data/input/proteomes/)')
    argparser.add_argument("--preference_sampling_method", type=str, help='how to sample the preference examples')
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--seed", type=int, default=None, help='RNG seed for replicate runs. Default None = unseeded (the historical behaviour: training was fully nondeterministic, and seed-less runs keep their existing model hash). Passing a seed both makes the run reproducible AND puts the seed in the model hash, so replicates get distinct model directories instead of colliding.')
    argparser.add_argument("--multithreading", action='store_true', help="should multithreading be used for the dataloader (can cause issues)")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=1, help="save model weights every n epochs")

    # continue training
    argparser.add_argument("--cont_ckpt_id", type=str, help="should training be continued? (model_id:ckpt_id)")

    # for hyperparameter search
    argparser.add_argument("--hyp", type=str, default=None, help="hyperparameter file")
    argparser.add_argument("--hyp_n", type=int, default=None, help="number of hyperparameter to train")

    try:
        args, args_unknown = kit.init('CAPE', 'CAPE-MPNN', create_job=True, arg_string=arg_string, argparser=argparser)
        main(args, args_unknown)

    except Exception as e:
        log_info("Exception: {e}")

        if G.ENV is None or "CONFIGS" not in G.ENV or G.ENV.CONFIGS.PM:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            kit.pdb.post_mortem(tb)

