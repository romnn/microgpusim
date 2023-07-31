use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    #[must_use]
    #[inline]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    #[inline]
    pub fn size(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }

    #[must_use]
    #[inline]
    pub fn into_tuple(&self) -> (u32, u32, u32) {
        (self.x, self.y, self.z)
    }
}

impl std::fmt::Display for Dim {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

impl From<nvbit_model::Dim> for Dim {
    fn from(dim: nvbit_model::Dim) -> Self {
        let nvbit_model::Dim { x, y, z } = dim;
        Self { x, y, z }
    }
}

impl From<u32> for Dim {
    #[inline]
    fn from(dim: u32) -> Self {
        Self { x: dim, y: 1, z: 1 }
    }
}

impl From<(u32, u32)> for Dim {
    #[inline]
    fn from(dim: (u32, u32)) -> Self {
        let (x, y) = dim;
        Self { x, y, z: 1 }
    }
}

impl From<(u32, u32, u32)> for Dim {
    #[inline]
    fn from(dim: (u32, u32, u32)) -> Self {
        let (x, y, z) = dim;
        Self { x, y, z }
    }
}

impl PartialEq<Point> for Dim {
    fn eq(&self, other: &Point) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

pub fn accelsim_block_id(block_id: &Dim, grid: &Dim) -> u64 {
    let block_x = block_id.x as u64;
    let block_y = block_id.y as u64;
    let block_z = block_id.z as u64;

    let grid_x = grid.x as u64;
    let grid_y = grid.y as u64;

    // tb_id = tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
    block_z * grid_y * grid_x + block_y * grid_x + block_x
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Point {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub bounds: Dim,
}

impl std::fmt::Display for Point {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

impl Point {
    pub fn new(point: Dim, bounds: Dim) -> Self {
        let Dim { x, y, z } = point;
        Self { x, y, z, bounds }
    }

    pub fn to_dim(&self) -> Dim {
        let Self { x, y, z, .. } = self.clone();
        Dim { x, y, z }
    }

    pub fn id(&self) -> u64 {
        let Self { x, y, z, bounds } = self;
        u64::from(*x)
            + u64::from(bounds.x) * u64::from(*y)
            + u64::from(bounds.x) * u64::from(bounds.y) * u64::from(*z)
    }

    pub fn accelsim_id(&self) -> u64 {
        // let Self { x, y, z, bounds } = self;
        accelsim_block_id(&self.to_dim(), &self.bounds)
        // let grid_x = bounds.x as u64;
        // let grid_y = bounds.y as u64;
        //
        // // tb_id = tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
        // u64::from(*z) * grid_y * grid_x + u64::from(*y) * grid_x + u64::from(*x)
    }
}

/// Iterates over 3-dimensional coordinates.
#[derive(Debug, Clone)]
pub struct Iter {
    bounds: Dim,
    current: u64,
}

impl Iter {
    #[must_use]
    #[inline]
    pub fn size(&self) -> u64 {
        self.bounds.size()
    }
}

impl Iterator for Iter {
    type Item = Point;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_lossless)]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { current, bounds } = self;
        if *current >= bounds.size() {
            return None;
        }
        let x = *current / (bounds.y * bounds.z) as u64;
        let yz = *current % (bounds.y * bounds.z) as u64;
        let y = yz / bounds.z as u64;
        let z = yz % bounds.z as u64;
        self.current += 1;
        Some(Point {
            x: x as u32,
            y: y as u32,
            z: z as u32,
            bounds: bounds.clone(),
        })
    }
}

impl IntoIterator for Dim {
    type Item = Point;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            bounds: self,
            current: 0,
        }
    }
}

// impl Ord for Dim {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         // self.as_tuple().cmp(&other.as_tuple())
//         todo!();
//     }
// }
//
// impl PartialOrd for Dim {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }

//   bool no_more_ctas_to_run() const {
//     return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
//             m_next_cta.z >= m_grid_dim.z);
//   }
//
// this is the default tuple sorting but its only used for stream vector
// bool operator()(const dim3 &a, const dim3 &b) const {
//     if (a.z < b.z)
//       return true;
//     else if (a.y < b.y)
//       return true;
//     else if (a.x < b.x)
//       return true;
//     else
//       return false;
//   }
//

// void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
//   i.x++;
//   if (i.x >= bound.x) {
//     i.x = 0;
//     i.y++;
//     if (i.y >= bound.y) {
//       i.y = 0;
//       if (i.z < bound.z) i.z++;
//     }
//   }
// }

#[cfg(test)]
mod tests {
    use super::{Dim, Point};
    use similar_asserts as diff;

    #[test]
    fn test_block_id() {
        let grid = Dim { x: 2, y: 2, z: 2 };
        let block = Dim { x: 1, y: 0, z: 0 };
        diff::assert_eq!(have: Point::new(block, grid.clone()).id(), want: 1);
    }

    #[test]
    fn test_block_sorting() {
        let grid = Dim { x: 3, y: 4, z: 2 };
        let mut blocks: Vec<_> = grid.into_iter().collect();
        blocks.sort_by_key(|block| block.accelsim_id());
        let blocks: Vec<_> = blocks.iter().map(|p| p.to_dim().into_tuple()).collect();
        dbg!(&blocks);
        diff::assert_eq!(
            have: blocks,
            want: vec![
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (2, 1, 0),
                (0, 2, 0),
                (1, 2, 0),
                (2, 2, 0),
                (0, 3, 0),
                (1, 3, 0),
                (2, 3, 0),
                (0, 0, 1),
                (1, 0, 1),
                (2, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
                (2, 1, 1),
                (0, 2, 1),
                (1, 2, 1),
                (2, 2, 1),
                (0, 3, 1),
                (1, 3, 1),
                (2, 3, 1),
            ]
        );
    }
}