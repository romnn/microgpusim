use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Dim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to parse {value:?}: {source:?}")]
    Parse {
        value: String,
        source: Option<std::num::ParseIntError>,
    },
}

static DIM_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$").unwrap());

impl TryFrom<&str> for Dim {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let captures = DIM_REGEX.captures(&value).ok_or_else(|| Error::Parse {
            value: value.to_string(),
            source: None,
        })?;
        let get_dim = |i: usize| {
            let dim = captures
                .get(i)
                .ok_or_else(|| Error::Parse {
                    value: value.to_string(),
                    source: None,
                })?
                .as_str();
            dim.parse().map_err(|err| Error::Parse {
                value: value.to_string(),
                source: Some(err),
            })
        };

        Ok(Self {
            x: get_dim(1)?,
            y: get_dim(2)?,
            z: get_dim(3)?,
        })
    }
}

impl std::str::FromStr for Dim {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::try_from(value)
    }
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

impl From<Point> for Dim {
    #[inline]
    fn from(point: Point) -> Self {
        point.to_dim()
    }
}

impl From<nvbit_model::Dim> for Dim {
    #[inline]
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

#[must_use]
#[inline]
pub fn accelsim_block_id(block_id: &Dim, grid: &Dim) -> u64 {
    let block_x = u64::from(block_id.x);
    let block_y = u64::from(block_id.y);
    let block_z = u64::from(block_id.z);

    let grid_x = u64::from(grid.x);
    let grid_y = u64::from(grid.y);

    // tb_id = tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
    block_z * grid_y * grid_x + block_y * grid_x + block_x
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Point {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub bounds: Dim,
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

impl Point {
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(point: Dim, bounds: Dim) -> Self {
        let Dim { x, y, z } = point;
        Self { x, y, z, bounds }
    }

    #[must_use]
    pub fn to_dim(&self) -> Dim {
        let Self { x, y, z, .. } = self.clone();
        Dim { x, y, z }
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        let Self { x, y, z, bounds } = self;
        u64::from(*x)
            + u64::from(bounds.x) * u64::from(*y)
            + u64::from(bounds.x) * u64::from(bounds.y) * u64::from(*z)
    }

    #[must_use]
    pub fn accelsim_id(&self) -> u64 {
        accelsim_block_id(&self.to_dim(), &self.bounds)
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

#[cfg(test)]
mod tests {
    use super::{Dim, Point};
    use similar_asserts as diff;

    #[test]
    fn test_block_id() {
        let grid = Dim { x: 2, y: 2, z: 2 };
        let block = Dim { x: 1, y: 0, z: 0 };
        diff::assert_eq!(have: Point::new(block, grid).id(), want: 1);
    }

    #[test]
    fn test_block_sorting() {
        let grid = Dim { x: 3, y: 4, z: 2 };
        let mut blocks: Vec<_> = grid.into_iter().collect();
        blocks.sort_by_key(super::Point::accelsim_id);
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
