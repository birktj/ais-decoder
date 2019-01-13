extern crate hound;
extern crate gnuplot;
extern crate crc;

use std::process::{Command, Stdio};
use std::io::{Read, Write};

use std::collections::VecDeque;

use crc::crc16;

trait Signal {
    fn sample_from_now(&mut self, t: f64) -> f64;
    fn duration(&self) -> Option<f64>;

    fn sample_iter(self, dt: f64) -> SampleIter<Self> where Self: Sized {
        SampleIter {
            s: self,
            dt,
        }
    }

    fn samples(&mut self, n: usize, dt: f64) -> Vec<f64> {
        let mut res = Vec::with_capacity(n);

        for _ in 0..n {
            res.push(self.sample_from_now(dt));
        }

        res
    }

    fn map<F: Fn(f64) -> f64>(self, f: F) -> MappedSignal<Self, F> where Self: Sized { 
        MappedSignal {
            s: self,
            f,
        }
    }
}

#[derive(Clone)]
struct SampleIter<S> {
    dt: f64,
    s: S,
}

impl<S: Signal> Iterator for SampleIter<S> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        Some(self.s.sample_from_now(self.dt))
    }
}

#[derive(Clone)]
struct MappedSignal<S, F> {
    s: S,
    f: F,
}

impl<F: Fn(f64) -> f64, S: Signal> Signal for MappedSignal<S, F> {
    fn duration(&self) -> Option<f64> {
        self.s.duration()
    }

    fn sample_from_now(&mut self, t: f64) -> f64 {
        (self.f)(self.s.sample_from_now(t))
    }
}

#[derive(Clone)]
struct FunctionSignal<F> {
    f: F,
    t0: f64,
}

impl<F: Fn(f64) -> f64> FunctionSignal<F> {
    fn new(f: F) -> FunctionSignal<F> {
        FunctionSignal {
            f,
            t0: 0.0,
        }
    }
}

impl<F: Fn(f64) -> f64> Signal for FunctionSignal<F> {
    fn duration(&self) -> Option<f64> {
        None
    }

    fn sample_from_now(&mut self, t: f64) -> f64 {
        self.t0 += t;
        (self.f)(self.t0)
    }
}

#[derive(Clone)]
struct SampledSignal<I> {
    sample_rate: f64,
    sample_count: Option<u64>,
    last_samples: [f64; 2],
    time_since_sample: f64,
    samples: I,
}

impl SampledSignal<()> {
    fn from_wav<R: std::io::Read>(reader: hound::WavReader<R>) -> SampledSignal<impl Iterator<Item = f64>> {
        let sample_rate = reader.spec().sample_rate as f64;
        let sample_count = Some(reader.len() as u64);
        
        eprintln!("Sample rate: {}", sample_rate);

        /*
        let samples = match reader.spec().sample_format {
            hound::SampleFormat::Int => Box::new(reader.into_samples::<i32>().map(|x| x.unwrap() as f64)) as Box<dyn Iterator<Item = f64>>,
            hound::SampleFormat::Float => Box::new(reader.into_samples::<f32>().map(|x| x.unwrap() as f64)) as Box<dyn Iterator<Item = f64>>,
        };
        */

        // FIXME: what about floats
        let mut samples = reader.into_samples::<i32>().map(|x| x.unwrap() as f64);

        SampledSignal {
            sample_rate,
            sample_count,
            last_samples: [samples.next().unwrap_or(0.0), samples.next().unwrap_or(0.0)],
            time_since_sample: 0.0,
            samples
        }
    }

    fn from_slice<'a>(sample_rate: f64, slice: &'a [f64]) -> SampledSignal<impl Iterator<Item = f64> + 'a> {
        let sample_count = Some(slice.len() as u64);

        let mut samples = slice.iter().map(|x| *x);

        SampledSignal {
            sample_rate,
            sample_count,
            last_samples: [samples.next().unwrap_or(0.0), samples.next().unwrap_or(0.0)],
            time_since_sample: 0.0,
            samples
        }
    }
}

impl<I: Iterator<Item = f64>> SampledSignal<I> {
    fn from_iterator(sample_rate: f64, mut samples: I) -> SampledSignal<I> {
        SampledSignal {
            sample_rate,
            sample_count: None,
            last_samples: [samples.next().unwrap_or(0.0), samples.next().unwrap_or(0.0)],
            time_since_sample: 0.0,
            samples
        }
    }

    fn to_owned(self) -> SampledSignal<std::vec::IntoIter<f64>> {
        let samples = self.samples.collect::<Vec<_>>().into_iter();
        SampledSignal {
            sample_rate: self.sample_rate,
            sample_count: self.sample_count,
            last_samples: self.last_samples,
            time_since_sample: self.time_since_sample,
            samples
        }
    }
}

impl<I: Iterator<Item = f64>> Signal for SampledSignal<I> {
    fn duration(&self) -> Option<f64> {
        self.sample_count.map(|cnt| cnt as f64 / self.sample_rate)
    }

    fn sample_from_now(&mut self, mut t: f64) -> f64 {
        let dt = 1.0 / self.sample_rate;

        while dt - self.time_since_sample < t {
            self.last_samples[0] = self.last_samples[1];
            self.last_samples[1] = self.samples.next().unwrap_or(self.last_samples[1]);
            t -= dt - self.time_since_sample;
            self.time_since_sample = 0.0;
        }

        self.time_since_sample += t;

        self.last_samples[0] * (1.0 - self.time_since_sample / dt) + self.last_samples[1] * (self.time_since_sample / dt)
    }
}

fn low_pass(x: &[f32]) -> Vec<f32> {
    let mut y = Vec::new();
    let a = 0.5;
    
    y.push(x[0] * a);

    for i in 1..x.len() {
        let v = a * x[i] + (1.0-a) * y[i-1];
        y.push(v);
    }
    y
}

fn low_pass_iter<I: Iterator<Item = f64>>(a: f64, mut iter: I) -> impl Iterator<Item = f64> {
    let y0 = iter.next().unwrap() * a; // FIXME
    iter.scan(y0, move |y, x| {
        *y = a * x + (1.0 - a) * (*y);
        Some(*y)
    })
}

const MSG_START: [bool; 8] = [false,true,true,true,true,true,true,false];
/*
fn ascii_6bit_encode(bytes: &[u8]) -> String {
    
}*/

struct AISdecoder {}

impl AISdecoder {
    fn bit(bytes: &[u8], bit: usize) -> bool {
        // TODO: error handling
        bytes[bit / 8] & (1<<(7 - bit%8)) != 0
    }

    fn unsigned_integer(bytes: &[u8], start: usize, end: usize) -> u64 {
        // TODO: optimize
        let mut res = 0;
        for i in start..end+1 {
            res <<= 1;
            res |= AISdecoder::bit(bytes, i) as u64;
        }
        res
    }

    fn signed_integer(bytes: &[u8], start: usize, end: usize) -> i64 {
        let res = AISdecoder::unsigned_integer(bytes, start+1, end) as i64;
        if AISdecoder::bit(bytes, start) {
            res * -1
        }
        else {
            res
        }
    }

    fn unsigned_float(precision: u32, bytes: &[u8], start: usize, end: usize) -> f64 {
        AISdecoder::unsigned_integer(bytes, start, end) as f64 / (10.0 * precision as f64)
    }

    fn signed_float(precision: u32, bytes: &[u8], start: usize, end: usize) -> f64 {
        AISdecoder::signed_integer(bytes, start, end) as f64 / (10.0 * precision as f64)
    }

    fn boolean(bytes: &[u8], pos: usize) -> bool {
        AISdecoder::bit(bytes, pos)
    }

    fn string(bytes: &[u8], start: usize, end: usize) -> String {
        let mut res = String::new();
        for i in (start..end).step_by(6) {
            let byte = AISdecoder::unsigned_integer(bytes, i, i+6);
            if byte < 32 {
                res.push((byte + 64) as u8 as char);
            }
            else {
                res.push((byte + 32) as u8 as char);
            }
        }
        res
    }
}

#[derive(Clone, Debug)]
struct NavigationBlock {
    repeat: u8,
    mmsi: u32,
    status: u8,
    turn: f64,
    speed: f64,
    accuracy: bool,
    lon: f64,
    lat: f64,
    course: f64,
    heading: u32,
    second: u8,
    maneuver: u8,
    raim: bool,
    radio: u32,
}

impl NavigationBlock {
    fn parse(data: &[u8]) -> NavigationBlock {
        NavigationBlock {
            repeat: AISdecoder::unsigned_integer(data, 6, 7) as u8,
            mmsi: AISdecoder::unsigned_integer(data, 8, 37) as u32,
            status: AISdecoder::unsigned_integer(data, 38, 41) as u8,
            turn: AISdecoder::signed_float(3, data, 42, 49),
            speed: AISdecoder::unsigned_float(1, data, 50, 59),
            accuracy: AISdecoder::boolean(data, 60),
            lon: AISdecoder::signed_float(4, data, 61, 88),
            lat: AISdecoder::signed_float(4, data, 89, 115),
            course: AISdecoder::unsigned_float(1, data, 116, 127),
            heading: AISdecoder::unsigned_integer(data, 128, 136) as u32,
            second: AISdecoder::unsigned_integer(data, 137, 142) as u8,
            maneuver: AISdecoder::unsigned_integer(data, 143, 144) as u8,
            raim: AISdecoder::boolean(data, 148),
            radio: AISdecoder::unsigned_integer(data, 149, 167) as u32,
        }
    }
}

fn decode_ais_frame(frame: &VecDeque<bool>) -> Option<Vec<u8>> {
    /*
    if &frame[0..16] != &MSG_START {
        return None
    }*/

    if frame.iter().take(8).ne(MSG_START.iter().take(8)) {
        return None
    }

    //eprintln!("Found possible message");
    //eprintln!("{}", frame.iter().map(|x| if *x {'1'} else {'0'}).collect::<String>());

    //eprintln!("Found message: ");

    let mut bits = Vec::new();
    let mut n_ones = 0;

    for b in frame.iter().skip(8) {
        if *b {
            if n_ones == 6 {
                return None
            }
            bits.push(*b);
            n_ones += 1;
        }
        else {
            if n_ones == 6 {
                bits.push(*b);
                break
            }
            else if n_ones < 5 {
                bits.push(*b);
            }
            n_ones = 0;
        }
    }

    /*
    let bits2 = frame.iter()
        .skip(16)
        .scan(0, |n_ones, &b| {
            if b {
                if *n_ones == 6 {
                    None
                }
                else {
                    *n_ones += 1;
                    Some(b)
                }
            }
            else {
                 
            }
        });*/
    
    let mut bytes = Vec::new();
    for i in (0..bits.len()).step_by(8) {
        let mut byte = 0u8;
        for j in i..std::cmp::min(bits.len(), i+8) {
            if bits[j] {
                byte |= 1 << (j - i);
            }
        }
        bytes.push(byte);
        //eprint!("{:02x}", byte);
    }
    //eprintln!("");

    let mut ascii_encoded = String::new();
    for i in (0..bits.len()).step_by(6) {
        let mut byte = 0u8;
        for j in i..std::cmp::min(bits.len(), i+6) {
            if bits[j] {
                byte |= 1 << (5 - (j - i));
            }
        }
        //eprint!("{} ", byte);
        if byte < 40 {
            ascii_encoded.push((byte + 48) as char);
        }
        else {
            ascii_encoded.push((byte + 56) as char);
        }
    }


    if bytes.len() < 3 {
        return None
    }

    let message_checksum: u16 = ((bytes[bytes.len()-2] as u16) << 8) + bytes[bytes.len()-3] as u16;
    let crc = crc16::checksum_x25(&bytes[0..bytes.len()-3]);

    if crc != message_checksum {
        eprintln!("Corrupted message");
        return None
    }

    eprintln!("Found message:");
    eprintln!("{}", ascii_encoded);

    let msg_type = AISdecoder::unsigned_integer(&bytes, 0, 5) as u8;

    if msg_type > 0 && msg_type < 4 {
        eprintln!("Navigation block: ");
        eprintln!("{:?}", NavigationBlock::parse(&bytes));
    }

    Some(bytes)
}

fn main() {
    //let reader = hound::WavReader::open("rtl-fm-test.wav").unwrap();
    //let reader = hound::WavReader::open("/home/birk/gqrx_20190113_215506_161975000.wav").unwrap();
    //let reader = hound::WavReader::open("/home/birk/recording.wav").unwrap();
    //let reader = hound::WavReader::open("fm-decode-long-test.wav").unwrap();
    //let reader = hound::WavReader::open("fm-decode-test.wav").unwrap();
    
    /*
    let sample_rate = reader.spec().sample_rate;

    eprintln!("Samplerate: {}", sample_rate);

    let samples = reader
        .samples::<i16>()
        .map(|x| x.map(|x| x as f32))
        .collect::<hound::Result<Vec<_>>>()
        .unwrap();

    let bin_samples = samples.iter().map(|s| if *s >= 0.0 {100.0} else {-100.0}).collect::<Vec<_>>();

    let samples2 = low_pass(&bin_samples);
    */
    let stdin = std::io::stdin();
    let data = stdin.lock()
        .bytes()
        .map(|b| b.unwrap())
        .scan(None, |state, b| {
            match *state {
                None => {
                    *state = Some(b);
                    Some(None)
                }
                Some(a) => {
                    *state = None;
                    Some(Some(((b as u16) << 8) | (a as u16)))
                }
            }
        })
        .flatten()
        .map(|x| x as i16 as f64)
        .map(|x| {
            //eprintln!("x: {}", x);
            x
        });

    //let mut signal = SampledSignal::from_wav(reader).to_owned();
    let mut signal = SampledSignal::from_iterator(50000.0, data);

    //let mut bin_signal = signal.clone()
    //    .map(|x| if x >= 0.0 {500.0} else {0.0});

    
    //let mut clock = FunctionSignal::new(|x| if ((x + 0.0009 * 3.0)*9600.0) as i64 % 2 >= 1 {250.0} else {0.0});
    let mut clock = FunctionSignal::new(|x| if (x*9600.0) as i64 % 2 >= 1 {250.0} else {0.0});

    /*
    let bit_signal = low_pass_iter(1.0, signal.clone().sample_iter(1.0 / (9600.0 * 20.0)))
        .take(16000)
        .scan(0.0, |s, x| {
            let res = f64::min(*s, x) < 0.0 && f64::max(*s, x) >= 0.0;
            *s = x;
            Some(res)
        })
        .scan(0, |s, x| {
            if x {
                let res = *s;
                *s = 0;
                Some(Some(res))
            }
            else if *s > 1000 {
                *s = 0;
                Some(Some(1000))
            }
            else {
                *s = *s + 1;
                Some(None)
            }
        })
        //.filter_map(|x| x)
        .flatten()
        //.filter(|x| *x >= 50)
        //.map(|x| {eprintln!("{}", x); x})
        .map(|x| std::cmp::max(1, (x + 7) / 20) - 1)
        .flat_map(|n| vec![std::iter::repeat(true).take(n), std::iter::repeat(false).take(1)].into_iter().flat_map(|x| x))
        .map(|x| if x {'1'} else {'0'})
        .collect::<String>();

    eprintln!("bits: {}", bit_signal);
    */

    let mut ais_data = low_pass_iter(1.0, signal.sample_iter(1.0 / (9600.0 * 20.0)))
        //.take(160000)
        .scan(0.0, |s, x| {
            let res = f64::min(*s, x) < 0.0 && f64::max(*s, x) >= 0.0;
            *s = x;
            Some(res)
        })
        .scan(0, |s, x| {
            if x {
                let res = *s;
                *s = 0;
                Some(Some(res))
            }
            else if *s > 1000 {
                *s = 0;
                Some(Some(1000))
            }
            else {
                *s = *s + 1;
                Some(None)
            }
        })
        //.filter_map(|x| x)
        .flatten()
        //.filter(|x| *x >= 50)
        //.map(|x| {eprintln!("{}", x); x})
        .map(|x| std::cmp::max(1, (x + 7) / 20) - 1)
        .flat_map(|n| vec![std::iter::repeat(true).take(n), std::iter::repeat(false).take(1)].into_iter().flat_map(|x| x))
        .scan(VecDeque::with_capacity(256), |state, b| {
            if state.len() >= 256 {
                state.pop_front();
            }
            state.push_back(b);
            //eprintln!("State: {:?}", state.iter().take(16).collect::<Vec<_>>());
            Some(decode_ais_frame(state))
        })
        .flatten();
        //.collect::<Vec<_>>();
        //.map(|x| x >= 0.0)
        //.map(|x| if x {'1'} else {'0'})
        //.collect::<String>();
    
    for frame in ais_data {
        eprintln!("Data: {:?}", frame);
    }

    /*
    let mut fig = gnuplot::Figure::new();
    fig.axes2d()
        .lines((0..), &signal.samples(4500, 0.00001), &[gnuplot::PlotOption::Color("#dddddd")])
        .lines((0..), &bin_signal.samples(4500, 0.00001), &[gnuplot::PlotOption::Color("#0000ff")])
        .lines((0..), &clock.samples(4500, 0.00001), &[gnuplot::PlotOption::Color("#00ff00")]);*/
        //.lines((0..), &samples2, &[gnuplot::PlotOption::Color("#0000ff")]);
    //fig.show();
    /*
    let mut plotter = Gnuplot::new();
    let data1 = plotter.write_1d_data(&samples);
    let data2 = plotter.write_1d_data(&bin_samples);
    plotter.write_command(&format!("plot {} with lines, {} with lines", data1, data2));
    //plotter.write_command(&format!("plot {} with lines", data2));


    plotter.plot();*/
    /*
    let p = Command::new("gnuplot")
        //.arg("-p")
        .arg("/dev/stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn().unwrap();

    let mut stdin = p.stdin.unwrap();

    fig.echo(&mut stdin);

    stdin.write_all(b"pause mouse close\n").unwrap();*/
}
