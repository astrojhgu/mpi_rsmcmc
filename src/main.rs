use std::fs::File;
use std::io::Write;


extern crate mpi;
extern crate mpi_rsmcmc;
extern crate rand;
use rand::thread_rng;

use mpi::topology::Communicator;
use mpi_rsmcmc::ensemble_sample::sample;
use mpi_rsmcmc::ptsample::sample as ptsample;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut ensemble = vec![
        vec![1., 2.],
        vec![3., 4.],
        vec![-1., -2.],
        vec![-1.5, -2.5],
        vec![0.0, 0.0],
        vec![0.1, -0.2],
        vec![0.5, 0.7],
        vec![0.6, 0.5],
    ];
    let mut old_lp: Vec<f64> = vec![];

    let mut rng = thread_rng();

    /*
    let (ensemble, old_lp) = sample(
        &mut |x: &Vec<f64>| -x.iter().map(|y| (*y).powi(2)).fold(0.0, |a, b| a + b),
        &(ensemble, old_lp),
        &mut rng,
        2.0,
        &world,
    ).unwrap();
    */

    let rank=world.rank();

    let aa=format!("log_{}.txt", rank);
    let mut beam_file = File::create(&aa).unwrap();


    for i in 0..100000{
        let xx=ptsample(&mut |x: &Vec<f64>| -x.iter().map(|y| (*y).powi(2)).fold(0.0, |a, b| a + b),
                                        &(ensemble, old_lp), &mut rng, &vec![1.0, 0.5], true, 2.0, &world).unwrap();
        ensemble=xx.0;
        old_lp=xx.1;
        for j in 0..2{
            writeln!(beam_file, "{} {}", ensemble[j][0], ensemble[j][1]);
        }
    }
}
