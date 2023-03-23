#[derive(Clone, Debug, Default)]
pub struct Dim {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Debug)]
pub struct DimIter {
    current: Dim,
    // pub x: usize,
    // pub y: usize,
    // pub z: usize,
}

impl Iterator for Counter {
    type Item = Dim;

    fn next(&mut self) -> Option<Self::Item> {
        let Dim {
            mut x,
            mut y,
            mut z,
        } = self.current;
        Some(current)
    }
}

// impl IntoIterator for Dim {
//     type Item = Dim;
//     type IntoIter = std::vec::IntoIter<Self::Item>;

//     fn into_iter(self) -> Self::IntoIter {
//         DimIter {}
//     }
// }

impl From<usize> for Dim {
    fn from(dim: usize) -> Self {
        Self { x: dim, y: 0, z: 0 }
    }
}

impl From<(usize, usize)> for Dim {
    fn from(dim: (usize, usize)) -> Self {
        let (x, y) = dim;
        Self { x, y, z: 0 }
    }
}

impl From<(usize, usize, usize)> for Dim {
    fn from(dim: (usize, usize, usize)) -> Self {
        let (x, y, z) = dim;
        Self { x, y, z }
    }
}
