#!/usr/bin/env bash

export PF="${CAPE_ROOT:?set CAPE_ROOT to the project root (see README)}/CAPE_MPNN"
export PYTHONPATH="${PF}/libs"
export PATH="${CAPE_ENV:?set CAPE_ENV to the cape_mpnn conda env}/bin:${PF}/external/programs/bin:${PF}/CAPE-MPNN:${PF}/tools:${PATH}"
