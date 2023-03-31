use std::io;

pub trait BufReadLine {
    fn read_line<'buf>(&mut self, buffer: &'buf mut String)
        -> Option<io::Result<&'buf mut String>>;
}

impl<R> BufReadLine for R
where
    R: io::BufRead,
{
    fn read_line<'buf>(
        &mut self,
        buffer: &'buf mut String,
    ) -> Option<io::Result<&'buf mut String>> {
        buffer.clear();

        io::BufRead::read_line(self, buffer)
            .map(|u| if u == 0 { None } else { Some(buffer) })
            .transpose()
    }
}
