use std::path::{Path, PathBuf};

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

#[inline]
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

#[inline]
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

#[inline]
pub fn remove_dir(path: impl AsRef<Path>) -> Result<(), Error> {
    let path = path.as_ref();
    std::fs::remove_dir_all(path).map_err(|source| Error::RemoveDirectory {
        path: path.to_path_buf(),
        source,
    })
}

#[inline]
pub fn create_dirs(path: impl AsRef<Path>) -> Result<(), Error> {
    use std::os::unix::fs::DirBuilderExt;
    let path = path.as_ref();
    match std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        // .mode(0o757)
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
