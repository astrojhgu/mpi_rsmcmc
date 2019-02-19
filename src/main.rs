#![allow(clippy::identity_op)]

use std::fs::File;
use std::io::Write;

extern crate mpi;
extern crate mpi_rsmcmc;
extern crate quickersort;
extern crate rand;
extern crate scorus;

use quickersort::sort_by;

use mpi::topology::Communicator;
use mpi_rsmcmc::ptsample::sample as ptsample;

use scorus::linear_space::type_wrapper::LsVec;

fn bimodal(x: &LsVec<f64, Vec<f64>>) -> f64 {
    if x[0] < -15.0 || x[0] > 15.0 || x[1] < 0.0 || x[1] > 1.0 {
        return -std::f64::INFINITY;
    }

    let (mu, sigma) = if x[1] < 0.5 { (-5.0, 0.1) } else { (5.0, 1.0) };

    -(x[0] - mu) * (x[0] - mu) / (2.0 * sigma * sigma) - sigma.ln()
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let mut outfile = File::create(&format!("data_{}.qdp", rank)).unwrap();

    let x: Vec<_> = vec![
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
        vec![0.10, 0.20],
        vec![0.20, 0.10],
        vec![0.23, 0.21],
        vec![0.03, 0.22],
        vec![0.10, 0.20],
        vec![0.20, 0.24],
        vec![0.20, 0.12],
        vec![0.23, 0.12],
    ].into_iter()
    .map(LsVec)
    .collect();
    let y = vec![0.0];
    let mut rng = rand::thread_rng();
    //let mut rng = rand::StdRng::new().unwrap();

    //let aa=(x,y);
    //let mut x=shuffle(&x, &mut rng);

    let blist = vec![
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015_625,
        0.007_812_5,
    ];
    let nbeta = blist.len();
    let nwalkers = x.len() / nbeta;
    let mut results: Vec<Vec<f64>> = Vec::new();
    let niter = 100_000;
    for i in 0..nbeta {
        results.push(Vec::new());
        results[i].reserve(niter);
    }

    let mut xy = (x, y);

    for k in 0..niter {
        //let aaa = ff(foo, &(x, y), &mut rng, 2.0, 1);

        let aa = ptsample(&bimodal, &xy, &mut rng, &blist, k % 10 == 0, 2.0, &world);
        xy = aa.unwrap();

        for (i, res) in results.iter_mut().enumerate().take(nbeta) {
            res.push(xy.0[i * nwalkers + 0][0]);
        }
    }

    for res in results.iter_mut().take(nbeta) {
        sort_by(res, &|x, y| {
            if x > y {
                std::cmp::Ordering::Greater
            } else if x < y {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        });
        for (j, rj) in res.iter().enumerate() {
            writeln!(outfile, "{} {}", rj, j).unwrap();
        }
        writeln!(outfile, "no no no").unwrap();
    }
}
