#![allow(clippy::many_single_char_names)]
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use num_traits::NumCast;
use std;
use std::ops::IndexMut;
use std::ops::{Add, Mul, Sub};

use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
//use std::sync::Arc;
use mpi::collective::CommunicatorCollectives;
use mpi::collective::Root;
use mpi::datatype::BufferMut;
use mpi::datatype::Equivalence;
use mpi::topology::Rank;

use scorus::linear_space::LinearSpace;
use scorus::mcmc::mcmc_errors::McmcErr;
use scorus::mcmc::utils::draw_z;
use scorus::mcmc::utils::scale_vec;
use scorus::utils::HasLen;
use scorus::utils::Resizeable;

pub fn sample<T, U, V, W, X, F, C>(
    flogprob: &F,
    ensemble_logprob: &(W, X),
    rng: &mut U,
    a: T,
    comm: &C,
) -> Result<(W, X), McmcErr>
where
    T: Float + NumCast + std::cmp::PartialOrd + SampleUniform + std::fmt::Display + Equivalence,
    Standard: Distribution<T>,
    U: Rng,
    V: Clone + LinearSpace<T> + AsMut<[T]>,
    for<'a> &'a V: Add<Output = V>,
    for<'a> &'a V: Sub<Output = V>,
    for<'a> &'a V: Mul<T, Output = V>,
    W: Clone + IndexMut<usize, Output = V> + HasLen,
    X: Clone
        + IndexMut<usize, Output = T>
        + HasLen
        + Resizeable<ElmType = T>
        + AsRef<[T]>
        + AsMut<[T]>,
    F: Fn(&V) -> T + ?Sized,
    C: CommunicatorCollectives,
    [T]: BufferMut,
{
    let rank = comm.rank();
    let root_rank = 0;
    let root_process = comm.process_at_rank(root_rank);

    let (ref ensemble, ref cached_logprob) = *ensemble_logprob;
    //    let cached_logprob = &ensemble_logprob.1;
    let mut result_ensemble = ensemble.clone();
    let mut result_logprob = cached_logprob.clone();
    //let pflogprob=Arc::new(&flogprob);
    let nwalkers = ensemble.len();

    if nwalkers == 0 {
        return Err(McmcErr::NWalkersIsZero);
    }

    if nwalkers % 2 != 0 {
        return Err(McmcErr::NWalkersIsNotEven);
    }

    let ndims: T = NumCast::from(ensemble[0].dimension()).unwrap();

    let half_nwalkers = nwalkers / 2;
    let mut walker_group: Vec<Vec<usize>> = vec![Vec::new(), Vec::new()];
    walker_group[0].reserve(half_nwalkers);
    walker_group[1].reserve(half_nwalkers);
    let mut walker_group_id: Vec<usize> = Vec::new();
    walker_group_id.reserve(nwalkers);
    let mut rvec: Vec<T> = Vec::new();
    let mut jvec: Vec<usize> = Vec::new();
    let mut zvec: Vec<T> = Vec::new();
    rvec.reserve(nwalkers);
    jvec.reserve(nwalkers);
    zvec.reserve(nwalkers);
    for i in 0..nwalkers {
        let mut gid: usize = rng.gen_range(0, 2);

        if walker_group[gid].len() == half_nwalkers {
            gid = 1 - gid;
        }
        walker_group[gid].push(i);
        walker_group_id.push(gid);
        rvec.push(rng.gen_range(zero::<T>(), one::<T>()));
        jvec.push(rng.gen_range(0, half_nwalkers));
        zvec.push(draw_z(rng, a));
    }

    root_process.broadcast_into(AsMut::<[usize]>::as_mut(&mut jvec));
    root_process.broadcast_into(AsMut::<[T]>::as_mut(&mut zvec));
    root_process.broadcast_into(AsMut::<[T]>::as_mut(&mut rvec));
    root_process.broadcast_into(AsMut::<[usize]>::as_mut(&mut walker_group_id));
    for wg in &mut walker_group {
        root_process.broadcast_into(AsMut::<[usize]>::as_mut(wg));
    }

    let lp_cached = result_logprob.len() == nwalkers;

    if !lp_cached {
        result_logprob.resize(nwalkers, T::zero());
    }
    //let lp_cached=cached_logprob.len()!=0;

    let comm_size = comm.size() as usize;

    let ntasks_per_node = {
        let x = nwalkers / comm_size;
        if x * comm_size >= nwalkers {
            x
        } else {
            x + 1
        }
    };

    for k in (rank as usize * ntasks_per_node)..((rank + 1) as usize * ntasks_per_node) {
        if k >= nwalkers {
            break;
        }
        let lp_last_y = if !lp_cached {
            let yy1 = flogprob(&ensemble[k]);
            result_logprob[k] = yy1;
            yy1
        } else {
            cached_logprob[k]
        };

        let i = walker_group_id[k];
        let j = jvec[k];
        let ni = 1 - i;
        let z = zvec[k];
        let r = rvec[k];
        let new_y = scale_vec(&ensemble[k], &ensemble[walker_group[ni][j]], z);
        let lp_y = flogprob(&new_y);

        let q = ((ndims - one::<T>()) * (z.ln()) + lp_y - lp_last_y).exp();
        {
            if r <= q {
                result_ensemble[k] = new_y;
                result_logprob[k] = lp_y;
            }
        }
    }

    for k in 0..nwalkers {
        let temp_root_id = (k / ntasks_per_node) as Rank;
        let temp_root = comm.process_at_rank(temp_root_id);
        temp_root.broadcast_into(AsMut::<[T]>::as_mut(&mut result_ensemble[k]));
        temp_root.broadcast_into(&mut result_logprob[k]);
    }

    Ok((result_ensemble, result_logprob))
}
