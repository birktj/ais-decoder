struct Gnuplot {
    command: String,
    data_i: u32,
}

impl Gnuplot {
    fn new() -> Gnuplot {
        Gnuplot {
            command: String::new(),
            data_i: 0,
        }
    }

    fn write_command(&mut self, cmd: &str) {
        self.command += cmd;
        self.command += "\n";
    }

    fn write_1d_data<T: std::fmt::Display>(&mut self, data: &[T]) -> String {
        let i = self.data_i;
        self.write_command(&format!("$data{} <<EOF", i));
        self.data_i += 1;
        for x in data {
            self.write_command(&format!("{}", x));
        }
        self.write_command("EOF");
        format!("$data{}", self.data_i-1)
    }

    fn write_2d_data<T1: std::fmt::Display, T2: std::fmt::Display>(&mut self, data: &[(T1, T2)]) {
        for (x, y) in data {
            self.command += &format!("{} {}\n", x, y);
        }
        self.command += "end\n";
    }

    fn plot(mut self) {
        self.command += "pause mouse close\n";

        //eprintln!("Gnuplot cmd:\n{}", self.command);

        let p = Command::new("gnuplot")
            //.arg("-p")
            .arg("/dev/stdin")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn().unwrap();

        p.stdin.unwrap().write_all(self.command.as_bytes()).unwrap();
    }
}
