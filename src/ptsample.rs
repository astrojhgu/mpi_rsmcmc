use std::ops::IndexMut;
use std::cmp::PartialOrd;
use std::fmt::Display;

use num_traits::float::Float;
use num_traits::NumCast;
use num_traits::identities::{one, zero};

use rand::{Rand, Rng};
use rand::distributions::range::SampleRange;
//use std::sync::Arc;

use mpi::collective::CommunicatorCollectives;
use mpi_sys::MPI_Comm;
use mpi::collective::Root;
use mpi::datatype::BufferMut;
use mpi::topology::Rank;
use mpi::datatype::Equivalence;

use scorus::mcmc::mcmc_errors::McmcErr;
use scorus::mcmc::ptsample::swap_walkers;
use scorus::mcmc::utils::{draw_z, scale_vec};
use scorus::utils::{HasLen, ItemSwapable, Resizeable};

fn only_sample_st<T, U, V, W, X, F, C>(
    flogprob: &mut F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    a: T,
    comm: &C,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + Rand + PartialOrd + SampleRange + Display + Equivalence,
    U: Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLen + AsRef<[T]> + AsMut<[T]>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Drop + ItemSwapable,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + Resizeable<ElmType = T>
        + Drop
        + ItemSwapable
        + AsRef<[T]>
        + AsMut<[T]>,
    F: FnMut(&V) -> T,
    C: CommunicatorCollectives<Raw = MPI_Comm>,
    [T]: BufferMut,
{
    let rank = comm.rank();
    let root_rank = 0;
    let root_process = comm.process_at_rank(root_rank);

    let (ref ensemble, ref cached_logprob) = *ensemble_logprob;

    let mut result_ensemble = ensemble.clone();
    let mut result_logprob = cached_logprob.clone();

    let nbeta = beta_list.len();
    let nwalkers = ensemble.len() / nbeta;

    if nwalkers == 0 {
        return Err(McmcErr::NWalkersIsZero);
    }
    if nwalkers % 2 != 0 {
        return Err(McmcErr::NWalkersIsNotEven);
    }

    if nbeta * nwalkers != ensemble.len() {
        return Err(McmcErr::NWalkersMismatchesNBeta);
    }

    let ndims: T = NumCast::from(ensemble[0].len()).unwrap();

    let half_nwalkers = nwalkers / 2;
    let mut walker_group: Vec<Vec<Vec<usize>>> = Vec::new();
    walker_group.reserve(nbeta);
    let mut walker_group_id: Vec<Vec<usize>> = Vec::new();
    walker_group_id.reserve(nbeta);

    let mut rvec: Vec<Vec<T>> = Vec::new();
    let mut jvec: Vec<Vec<usize>> = Vec::new();
    let mut zvec: Vec<Vec<T>> = Vec::new();

    for i in 0..nbeta {
        walker_group.push(vec![Vec::new(), Vec::new()]);
        walker_group[i][0].reserve(half_nwalkers);
        walker_group[i][1].reserve(half_nwalkers);

        walker_group_id.push(Vec::new());
        walker_group_id[i].reserve(nwalkers);

        rvec.push(Vec::new());
        jvec.push(Vec::new());
        zvec.push(Vec::new());

        rvec[i].reserve(nwalkers);
        jvec[i].reserve(nwalkers);
        zvec[i].reserve(nwalkers);

        for j in 0..nwalkers {
            let mut gid: usize = rng.gen_range(0, 2);
            if walker_group[i][gid].len() == half_nwalkers {
                gid = 1 - gid;
            }
            walker_group[i][gid].push(j);
            walker_group_id[i].push(gid);
            rvec[i].push(rng.gen_range(zero(), one()));
            jvec[i].push(rng.gen_range(0, half_nwalkers));
            zvec[i].push(draw_z(rng, a));
        }

        root_process.broadcast_into(AsMut::<[usize]>::as_mut(&mut jvec[i]));
        root_process.broadcast_into(AsMut::<[T]>::as_mut(&mut zvec[i]));
        root_process.broadcast_into(AsMut::<[T]>::as_mut(&mut rvec[i]));
        root_process.broadcast_into(AsMut::<[usize]>::as_mut(&mut walker_group_id[i]));
        for wg in &mut walker_group[i] {
            root_process.broadcast_into(AsMut::<[usize]>::as_mut(wg));
        }
    }

    let lp_cached = result_logprob.len() == result_ensemble.len();

    if !lp_cached {
        result_logprob.resize(result_ensemble.len(), T::zero());
    }
    //let lp_cached=cached_logprob.len()!=0;
    let comm_size = comm.size() as usize;

    let ntasks_per_node = {
        let x = (nwalkers * nbeta) / comm_size;
        if x * comm_size >= (nwalkers * nbeta) {
            x
        } else {
            x + 1
        }
    };

    for n in (rank as usize * ntasks_per_node)..((rank + 1) as usize * ntasks_per_node) {
        if n > nwalkers * nbeta {
            break;
        }

        let ibeta = n / nwalkers;
        let k = n - ibeta * nwalkers;
        let lp_last_y = match lp_cached {
            false => {
                //let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);
                let yy1 = flogprob(&ensemble[ibeta * nwalkers + k]);

                result_logprob[ibeta * nwalkers + k] = yy1;
                yy1
            }
            _ => cached_logprob[ibeta * nwalkers + k],
        };

        let i = walker_group_id[ibeta][k];
        let j = jvec[ibeta][k];
        let ni = 1 - i;
        let z = zvec[ibeta][k];
        let r = rvec[ibeta][k];
        let new_y = scale_vec(
            &ensemble[ibeta * nwalkers + k],
            &ensemble[ibeta * nwalkers + walker_group[ibeta][ni][j]],
            z,
        );
        let lp_y = flogprob(&new_y);
        let beta = beta_list[ibeta];
        let delta_lp = lp_y - lp_last_y;
        let q = ((ndims - one::<T>()) * (z.ln()) + delta_lp * beta).exp();

        if r <= q {
            result_ensemble[ibeta * nwalkers + k] = new_y;
            result_logprob[ibeta * nwalkers + k] = lp_y;
        }
    }

    for n in 0..(nwalkers * nbeta) {
        let temp_root_id = (n / ntasks_per_node) as Rank;
        let ibeta = n / nwalkers;
        let k = n - ibeta * nwalkers;
        let temp_root = comm.process_at_rank(temp_root_id);

        temp_root.broadcast_into(AsMut::<[T]>::as_mut(
            &mut result_ensemble[ibeta * nwalkers + k],
        ));
        temp_root.broadcast_into(&mut result_logprob[ibeta * nwalkers + k]);
    }

    Ok((result_ensemble, result_logprob))
}

pub fn sample<T, U, V, W, X, F, C>(
    flogprob: &mut F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    beta_list: &X,
    perform_swap: bool,
    a: T,
    comm: &C,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + Rand + PartialOrd + SampleRange + Display + Equivalence,
    U: Rng,
    V: Clone + IndexMut<usize, Output = T> + HasLen + AsRef<[T]> + AsMut<[T]>,
    W: Clone + IndexMut<usize, Output = V> + HasLen + Drop + ItemSwapable,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + Resizeable<ElmType = T>
        + Drop
        + ItemSwapable
        + AsRef<[T]>
        + AsMut<[T]>,
    F: FnMut(&V) -> T,
    C: CommunicatorCollectives<Raw = MPI_Comm>,
    [T]: BufferMut,
{
    let perform_swap = {
        let mut perform_swap = perform_swap;
        let root = comm.process_at_rank(0);
        root.broadcast_into(&mut perform_swap);
        perform_swap
    };
    if perform_swap {
        let mut ensemble_logprob1 = (ensemble_logprob.0.clone(), ensemble_logprob.1.clone());
        swap_walkers(&mut ensemble_logprob1, rng, beta_list)?;

        let root = comm.process_at_rank(0);
        for i in 0..ensemble_logprob1.0.len() {
            root.broadcast_into(AsMut::<[T]>::as_mut(&mut ensemble_logprob1.0[i]));
        }

        for i in 0..ensemble_logprob1.1.len() {
            root.broadcast_into(&mut ensemble_logprob1.1[i]);
        }

        only_sample_st(flogprob, &ensemble_logprob1, rng, beta_list, a, comm)
    } else {
        only_sample_st(flogprob, ensemble_logprob, rng, beta_list, a, comm)
    }
}
