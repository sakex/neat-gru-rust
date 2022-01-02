use futures::future::join_all;
use num::Float;
use std::fmt::Display;
use std::time::Duration;
use tokio::sync::mpsc::{channel, Receiver, Sender};

pub struct SpikingNeuron<T>
where
    T: Float + std::ops::AddAssign + Display + Send + Sync + 'static,
{
    /// Genetic can have threshold one > threshold 2 or the opposite
    pub threshold_one: T,
    pub threshold_two: T,
    pub decay: T,
    pub input: Receiver<T>,
    pub outputs: Vec<(T, Sender<T>)>,
}

impl<T> SpikingNeuron<T>
where
    T: Float + std::ops::AddAssign + Display + Send + Sync + 'static,
{
    pub fn new() -> (Self, Sender<T>) {
        let (sdr, rcv) = channel(1);
        (
            Self {
                threshold_one: T::zero(),
                threshold_two: T::zero(),
                decay: T::zero(),
                input: rcv,
                outputs: Vec::new(),
            },
            sdr,
        )
    }

    pub fn spawn_task(self) {
        let Self {
            threshold_one,
            threshold_two,
            decay,
            mut input,
            mut outputs,
        } = self;
        tokio::spawn(async move {
            let threshold_up = threshold_one.max(threshold_two);
            let threshold_down = threshold_one.min(threshold_two);
            let mid_point = (threshold_up - threshold_down) / T::from(2.0).unwrap();
            let mut activation: T = mid_point;
            loop {
                tokio::select! {
                    input = input.recv() => {
                        if let Some(input) = input {
                            activation += input;
                            if activation >= threshold_up || activation <= threshold_down {
                                let futures: Vec<_> = outputs.iter_mut().map( |(weight, sdr)| {
                                    sdr.send(*weight * activation)
                                }).collect();
                                join_all(futures).await;
                                activation = mid_point;
                            }
                        } else {
                            break;
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {
                        activation = activation - (activation - mid_point) * decay;
                    }
                }
            }
        });
    }
}
