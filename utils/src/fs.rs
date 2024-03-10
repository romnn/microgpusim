use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub usize);

impl std::fmt::Display for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", human_bytes::human_bytes(self.0 as f64))
    }
}

#[derive(thiserror::Error, Debug)]
pub enum InvalidSizeError {
    #[error(transparent)]
    Parse(#[from] parse_size::Error),

    #[error(transparent)]
    Cast(#[from] std::num::TryFromIntError),
}

impl std::str::FromStr for Bytes {
    type Err = InvalidSizeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let bytes: u64 = parse_size::parse_size(value)?;
        let bytes: usize = bytes.try_into()?;
        Ok(Self(bytes))
    }
}

pub fn multi_glob<I, S>(patterns: I) -> impl Iterator<Item = Result<PathBuf, glob::GlobError>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let globs = patterns.into_iter().map(|p| glob::glob(p.as_ref()));
    globs.flatten().flatten()
}

#[allow(clippy::permissions_set_readonly_false)]
pub fn rchmod_writable(root: impl AsRef<Path>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root.as_ref())? {
        let entry = entry?;
        let path = entry.path();
        path.metadata()?.permissions().set_readonly(false);
    }
    Ok(())
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("could not open file {path:?}")]
    OpenFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not set mode of {path:?} to {mode:o}")]
    SetPermissions {
        path: PathBuf,
        mode: u32,
        source: std::io::Error,
    },
    #[error("could not create directories {path:?}")]
    CreateDirectories {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not remove directory {path:?}")]
    RemoveDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl From<Error> for std::io::Error {
    fn from(err: Error) -> Self {
        match err {
            Error::OpenFile { source, .. }
            | Error::SetPermissions { source, .. }
            | Error::CreateDirectories { source, .. }
            | Error::RemoveDirectory { source, .. } => source,
        }
    }
}

// #[inline]
pub fn open_readable(path: impl AsRef<Path>) -> Result<std::io::BufReader<std::fs::File>, Error> {
    let path = path.as_ref();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|source| Error::OpenFile {
            source,
            path: path.to_path_buf(),
        })?;
    let reader = std::io::BufReader::new(file);
    Ok(reader)
}

// #[inline]
pub fn open_writable(path: impl AsRef<Path>) -> Result<std::io::BufWriter<std::fs::File>, Error> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mode: u32 = 0o777;
    let path = path.as_ref();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .mode(mode)
        .create(true)
        .open(path)
        .map_err(|source| Error::OpenFile {
            source,
            path: path.to_path_buf(),
        })?;

    let mut permissions = file
        .metadata()
        .map_err(|source| Error::OpenFile {
            source,
            path: path.to_path_buf(),
        })?
        .permissions();
    permissions.set_mode(mode);
    file.set_permissions(permissions)
        .map_err(|source| Error::SetPermissions {
            source,
            mode,
            path: path.to_path_buf(),
        })?;

    Ok(std::io::BufWriter::new(file))
}

// #[inline]
pub fn remove_dir(path: impl AsRef<Path>) -> Result<(), Error> {
    let path = path.as_ref();
    match std::fs::remove_dir_all(path) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(Error::RemoveDirectory {
            path: path.to_path_buf(),
            source,
        }),
        Ok(()) => Ok(()),
    }
}

// #[inline]
pub fn create_dirs(path: impl AsRef<Path>) -> Result<(), Error> {
    use std::os::unix::fs::DirBuilderExt;
    let path = path.as_ref();
    match std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(path)
    {
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(source) => Err(Error::CreateDirectories {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// Normalize paths
///
/// Unlike `std::fs::Path::canonicalize`, this function does not access the file system.
/// Hence, this function can be used for paths that do not (yet) exist.
///
/// # Source:
/// [cargo](https://github.com/rust-lang/cargo/blob/fede83ccf973457de319ba6fa0e36ead454d2e20/src/cargo/util/paths.rs#L61)
#[must_use]
pub fn normalize_path(path: impl AsRef<Path>) -> PathBuf {
    use std::path::Component;
    let mut components = path.as_ref().components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().copied() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::Normal(c) => {
                ret.push(c);
            }
        }
    }
    ret
}

pub trait PathExt {
    #[must_use]
    fn resolve<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>;

    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>;

    #[must_use]
    fn normalize(&self) -> PathBuf;

    fn try_normalize(&self) -> Result<PathBuf, std::io::Error>;
}

impl PathExt for Path {
    // #[inline]
    #[must_use]
    fn resolve<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        if self.is_absolute() {
            self.normalize()
        } else {
            base.as_ref().join(self).normalize()
        }
    }

    // #[inline]
    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        let rel_path: PathBuf =
            pathdiff::diff_paths(self, base).unwrap_or_else(|| self.to_path_buf());
        rel_path.normalize()
    }

    fn try_normalize(&self) -> Result<PathBuf, std::io::Error> {
        Ok(normalize_path(self))
    }

    // #[inline]
    #[must_use]
    fn normalize(&self) -> PathBuf {
        self.try_normalize().unwrap_or_else(|_| self.to_path_buf())
    }
}

#[allow(clippy::unnecessary_wraps)]
#[cfg(test)]
mod tests {
    use super::PathExt;
    use color_eyre::eyre;
    use std::path::PathBuf;

    #[test]
    fn test_path_normalize() -> eyre::Result<()> {
        diff::assert_eq!(
            have: PathBuf::from("/base/./vectoradd/vectoradd").try_normalize()?,
            want: PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff::assert_eq!(
            have: PathBuf::from("/base/../vectoradd/vectoradd").try_normalize()?,
            want: PathBuf::from("/vectoradd/vectoradd")
        );
        Ok(())
    }

    #[test]
    fn test_path_resolve_on_absolute_path() -> eyre::Result<()> {
        let absolute_path = PathBuf::from("/base/vectoradd/vectoradd");
        diff::assert_eq!(have: absolute_path.resolve("/another-base"), want: absolute_path);
        diff::assert_eq!(have: absolute_path.resolve("test"), want: absolute_path);
        diff::assert_eq!(have: absolute_path.resolve("../something"), want: absolute_path);
        diff::assert_eq!(have: absolute_path.resolve(""), want: absolute_path);
        Ok(())
    }

    #[test]
    fn test_path_resolve_absolute_base() -> eyre::Result<()> {
        diff::assert_eq!(
            have: PathBuf::from("./vectoradd/vectoradd").resolve("/base/"),
            want: PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff::assert_eq!(
            have: PathBuf::from("././vectoradd/vectoradd").resolve("/base"),
            want: PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff::assert_eq!(
            have: PathBuf::from("vectoradd/vectoradd").resolve("/base"),
            want: PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff::assert_eq!(
            have: PathBuf::from("vectoradd/vectoradd")
                .resolve("/base")
                .resolve("/base"),
            want: PathBuf::from("/base/vectoradd/vectoradd")
        );
        Ok(())
    }

    #[test]
    fn test_path_resolve_relative_base() -> eyre::Result<()> {
        diff::assert_eq!(
            have: PathBuf::from("./vectoradd/vectoradd").resolve("base/"),
            want: PathBuf::from("base/vectoradd/vectoradd")
        );
        diff::assert_eq!(
            have: PathBuf::from("././vectoradd/vectoradd").resolve("./base/test/"),
            want: PathBuf::from("base/test/vectoradd/vectoradd")
        );
        // at the moment, we do not guard against possibly unwanted behaviour when resolving
        // multiple times on the same relative path accidentally.
        diff::assert_eq!(
            have: PathBuf::from("vectoradd/vectoradd")
                .resolve("base")
                .resolve("base"),
            want: PathBuf::from("base/base/vectoradd/vectoradd")
        );
        Ok(())
    }
}
